# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.distributed as dist
import time

from fla.layers.utils import pad_input, unpad_input
from fla.layers.freq_analysis_utils import *
from fla.layers.router_norm import (
    RouterNorm,
    RouterNormStatsLogger,
    apply_router_norm_mode,
    build_router_norm_config,
)
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.path_attn.parallel import parallel_path_attn

import math
import random
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
if TYPE_CHECKING:
    from fla.models.utils import Cache
from rotary_embedding_torch import RotaryEmbedding
from transformers.activations import NewGELUActivation

import pdb
import os
import torch
from tqdm import tqdm

def _tensor_stats(x: torch.Tensor, name: str):
    if x is None:
        return {f"{name}_is_none": True}
    x_fp = x.detach()
    finite = torch.isfinite(x_fp)
    numel = x_fp.numel()
    n_finite = int(finite.sum().item())
    n_bad = int(numel - n_finite)
    # use finite values for stats to avoid nan pollution
    if n_finite > 0:
        xf = x_fp[finite]
        return {
            f"{name}_shape": tuple(x_fp.shape),
            f"{name}_dtype": str(x_fp.dtype),
            f"{name}_device": str(x_fp.device),
            f"{name}_numel": int(numel),
            f"{name}_n_bad": n_bad,
            f"{name}_mean": float(xf.mean().item()),
            f"{name}_std": float(xf.std(unbiased=False).item()),
            f"{name}_min": float(xf.min().item()),
            f"{name}_max": float(xf.max().item()),
        }
    else:
        return {
            f"{name}_shape": tuple(x_fp.shape),
            f"{name}_dtype": str(x_fp.dtype),
            f"{name}_device": str(x_fp.device),
            f"{name}_numel": int(numel),
            f"{name}_n_bad": n_bad,
        }

def _log_stats(stats: dict, prefix: str = "[rel_record]"):
    # keep it single-line-ish for grep
    msg = prefix + " " + " ".join([f"{k}={v}" for k, v in stats.items()])
    print(msg)


class RunningBinStats:
    """
    Streaming stats over query-position bins for eval-only diagnostics.
    """

    _SUM_KEYS = (
        "sum_mu_base",
        "sum_std_base",
        "sum_mu_rel",
        "sum_std_rel",
        "sum_r",
        "sum_kl",
        "sum_abs_base",
        "sum_abs_rel",
    )
    _SAMPLE_KEYS = (
        "samples_std_base",
        "samples_std_rel",
        "samples_r",
        "samples_kl",
    )

    def __init__(
        self,
        bin_size: int = 256,
        eps: float = 1e-6,
        per_head: bool = False,
        stats_dtype: torch.dtype = torch.float32,
        max_samples_per_bin: int = 4096,
    ):
        self.bin_size = max(1, int(bin_size))
        self.eps = float(eps)
        self.per_head = bool(per_head)
        self.stats_dtype = stats_dtype
        self.max_samples_per_bin = max(128, int(max_samples_per_bin))
        self._bins = {}
        self._vec_len = None

    def _new_bucket(self, vec_len: int):
        z = torch.zeros(vec_len, dtype=self.stats_dtype, device="cpu")
        return {
            "sum_mu_base": z.clone(),
            "sum_std_base": z.clone(),
            "sum_mu_rel": z.clone(),
            "sum_std_rel": z.clone(),
            "sum_r": z.clone(),
            "sum_kl": z.clone(),
            "sum_abs_base": z.clone(),
            "sum_abs_rel": z.clone(),
            "count": z.clone(),
            "samples_std_base": [],
            "samples_std_rel": [],
            "samples_r": [],
            "samples_kl": [],
            "seen_std_base": 0,
            "seen_std_rel": 0,
            "seen_r": 0,
            "seen_kl": 0,
        }

    def _ensure_bucket(self, bin_idx: int, vec_len: int):
        if self._vec_len is None:
            self._vec_len = int(vec_len)
        if self._vec_len != int(vec_len):
            raise ValueError(f"RunningBinStats vec_len mismatch: {self._vec_len} vs {vec_len}")
        bucket = self._bins.get(int(bin_idx))
        if bucket is None:
            bucket = self._new_bucket(vec_len)
            self._bins[int(bin_idx)] = bucket
        return bucket

    def _reservoir_add(self, bucket: dict, sample_key: str, seen_key: str, values: torch.Tensor):
        vals = values.detach().flatten().to(torch.float32).cpu().tolist()
        if not vals:
            return
        sample = bucket[sample_key]
        seen = int(bucket.get(seen_key, 0))
        cap = self.max_samples_per_bin
        for v in vals:
            seen += 1
            if len(sample) < cap:
                sample.append(float(v))
            else:
                j = random.randint(1, seen)
                if j <= cap:
                    sample[j - 1] = float(v)
        bucket[seen_key] = seen

    @torch.no_grad()
    def update(self, z_base: torch.Tensor, rel: torch.Tensor, coe_for_rel=None):
        if z_base is None or rel is None:
            return
        zb = z_base.detach().to(dtype=self.stats_dtype)
        rr = rel.detach().to(dtype=self.stats_dtype)
        if coe_for_rel is None:
            z_total = zb + rr
        else:
            if not torch.is_tensor(coe_for_rel):
                coe_for_rel = torch.tensor(float(coe_for_rel), device=zb.device, dtype=self.stats_dtype)
            else:
                coe_for_rel = coe_for_rel.detach().to(device=zb.device, dtype=self.stats_dtype)
            z_total = zb + coe_for_rel * rr

        if self.per_head:
            # [H, T]
            mu_base = zb.mean(dim=-1).mean(dim=0)
            std_base = zb.std(dim=-1, unbiased=False).mean(dim=0)
            mu_rel = rr.mean(dim=-1).mean(dim=0)
            std_rel = rr.std(dim=-1, unbiased=False).mean(dim=0)
            abs_base = zb.abs().mean(dim=-1).mean(dim=0)
            abs_rel = rr.abs().mean(dim=-1).mean(dim=0)
        else:
            # [T]
            mu_base = zb.mean(dim=-1).mean(dim=(0, 1))
            std_base = zb.std(dim=-1, unbiased=False).mean(dim=(0, 1))
            mu_rel = rr.mean(dim=-1).mean(dim=(0, 1))
            std_rel = rr.std(dim=-1, unbiased=False).mean(dim=(0, 1))
            abs_base = zb.abs().mean(dim=-1).mean(dim=(0, 1))
            abs_rel = rr.abs().mean(dim=-1).mean(dim=(0, 1))

        logp_base = F.log_softmax(zb, dim=-1)
        logp_total = F.log_softmax(z_total, dim=-1)
        p_base = logp_base.exp()
        kl_full = (p_base * (logp_base - logp_total)).sum(dim=-1)  # [B,H,T]
        if self.per_head:
            kl = kl_full.mean(dim=0)  # [H,T]
        else:
            kl = kl_full.mean(dim=(0, 1))  # [T]
        r = std_rel / (std_base + self.eps)

        if self.per_head:
            _, T = mu_base.shape
            vec_len = int(mu_base.shape[0])
        else:
            T = int(mu_base.shape[0])
            vec_len = 1

        for start in range(0, T, self.bin_size):
            end = min(start + self.bin_size, T)
            bidx = start // self.bin_size
            bucket = self._ensure_bucket(bidx, vec_len=vec_len)
            seg_len = end - start
            if seg_len <= 0:
                continue

            if self.per_head:
                bucket["sum_mu_base"] += mu_base[:, start:end].sum(dim=-1).cpu()
                bucket["sum_std_base"] += std_base[:, start:end].sum(dim=-1).cpu()
                bucket["sum_mu_rel"] += mu_rel[:, start:end].sum(dim=-1).cpu()
                bucket["sum_std_rel"] += std_rel[:, start:end].sum(dim=-1).cpu()
                bucket["sum_r"] += r[:, start:end].sum(dim=-1).cpu()
                bucket["sum_kl"] += kl[:, start:end].sum(dim=-1).cpu()
                bucket["sum_abs_base"] += abs_base[:, start:end].sum(dim=-1).cpu()
                bucket["sum_abs_rel"] += abs_rel[:, start:end].sum(dim=-1).cpu()
                bucket["count"] += float(seg_len)

                self._reservoir_add(bucket, "samples_std_base", "seen_std_base", std_base[:, start:end])
                self._reservoir_add(bucket, "samples_std_rel", "seen_std_rel", std_rel[:, start:end])
                self._reservoir_add(bucket, "samples_r", "seen_r", r[:, start:end])
                self._reservoir_add(bucket, "samples_kl", "seen_kl", kl[:, start:end])
            else:
                bucket["sum_mu_base"] += torch.tensor([mu_base[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_std_base"] += torch.tensor([std_base[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_mu_rel"] += torch.tensor([mu_rel[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_std_rel"] += torch.tensor([std_rel[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_r"] += torch.tensor([r[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_kl"] += torch.tensor([kl[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_abs_base"] += torch.tensor([abs_base[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["sum_abs_rel"] += torch.tensor([abs_rel[start:end].sum().item()], dtype=self.stats_dtype)
                bucket["count"] += float(seg_len)

                self._reservoir_add(bucket, "samples_std_base", "seen_std_base", std_base[start:end])
                self._reservoir_add(bucket, "samples_std_rel", "seen_std_rel", std_rel[start:end])
                self._reservoir_add(bucket, "samples_r", "seen_r", r[start:end])
                self._reservoir_add(bucket, "samples_kl", "seen_kl", kl[start:end])

    @staticmethod
    def _to_vec(value, vec_len: int):
        if isinstance(value, torch.Tensor):
            out = value.detach().float().cpu().view(-1)
            if out.numel() == vec_len:
                return out
            if out.numel() == 1 and vec_len > 1:
                return out.repeat(vec_len)
            return torch.zeros(vec_len, dtype=torch.float32)
        if isinstance(value, (list, tuple)):
            vals = [float(v) for v in value]
            if len(vals) == vec_len:
                return torch.tensor(vals, dtype=torch.float32)
            if len(vals) == 1 and vec_len > 1:
                return torch.full((vec_len,), float(vals[0]), dtype=torch.float32)
            return torch.zeros(vec_len, dtype=torch.float32)
        try:
            return torch.full((vec_len,), float(value), dtype=torch.float32)
        except Exception:
            return torch.zeros(vec_len, dtype=torch.float32)

    @classmethod
    def from_state_dict(cls, state: dict):
        if not isinstance(state, dict):
            return cls()
        obj = cls(
            bin_size=int(state.get("bin_size", 256)),
            eps=float(state.get("eps", 1e-6)),
            per_head=bool(state.get("per_head", False)),
            stats_dtype=torch.float32,
            max_samples_per_bin=int(state.get("max_samples_per_bin", 4096)),
        )
        vec_len = int(state.get("vec_len", 1))
        obj._vec_len = max(1, vec_len)
        bins = state.get("bins", {})
        if not isinstance(bins, dict):
            return obj
        for bidx_raw, src in bins.items():
            try:
                bidx = int(bidx_raw)
            except Exception:
                continue
            bucket = obj._new_bucket(obj._vec_len)
            for k in obj._SUM_KEYS + ("count",):
                bucket[k] = obj._to_vec(src.get(k, 0.0), obj._vec_len).to(dtype=torch.float32)
            for sk in obj._SAMPLE_KEYS:
                vals = src.get(sk, [])
                if isinstance(vals, (list, tuple)):
                    bucket[sk] = [float(v) for v in vals[: obj.max_samples_per_bin]]
                else:
                    bucket[sk] = []
                seen_k = "seen_" + sk[len("samples_") :]
                try:
                    bucket[seen_k] = int(src.get(seen_k, len(bucket[sk])))
                except Exception:
                    bucket[seen_k] = len(bucket[sk])
            obj._bins[bidx] = bucket
        return obj

    def merge(self, other):
        if isinstance(other, dict):
            other = RunningBinStats.from_state_dict(other)
        if not isinstance(other, RunningBinStats):
            return
        if other._vec_len is None:
            return
        if self._vec_len is None:
            self._vec_len = other._vec_len
        if self._vec_len != other._vec_len:
            return
        for bidx, src in other._bins.items():
            dst = self._ensure_bucket(bidx, self._vec_len)
            for k in self._SUM_KEYS + ("count",):
                dst[k] += src[k].to(dst[k].dtype)
            for sk in self._SAMPLE_KEYS:
                seen_k = "seen_" + sk[len("samples_") :]
                vals = src.get(sk, [])
                if vals:
                    self._reservoir_add(dst, sk, seen_k, torch.tensor(vals, dtype=torch.float32))

    def state_dict(self):
        out = {
            "bin_size": int(self.bin_size),
            "eps": float(self.eps),
            "per_head": bool(self.per_head),
            "max_samples_per_bin": int(self.max_samples_per_bin),
            "vec_len": int(self._vec_len or 1),
            "bins": {},
        }
        for bidx in sorted(self._bins.keys()):
            bucket = self._bins[bidx]
            bo = {}
            for k in self._SUM_KEYS + ("count",):
                bo[k] = bucket[k].detach().float().cpu().tolist()
            for sk in self._SAMPLE_KEYS:
                bo[sk] = [float(v) for v in bucket.get(sk, [])[: self.max_samples_per_bin]]
                seen_k = "seen_" + sk[len("samples_") :]
                bo[seen_k] = int(bucket.get(seen_k, len(bo[sk])))
            out["bins"][int(bidx)] = bo
        return out

    @staticmethod
    def _quantile_dict(values):
        if not values:
            return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
        t = torch.tensor(values, dtype=torch.float32)
        q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
        return {"p50": float(q[0].item()), "p90": float(q[1].item()), "p99": float(q[2].item())}

    def summary(self):
        out = {}
        for bidx in sorted(self._bins.keys()):
            b = self._bins[bidx]
            c = b["count"].clamp_min(1.0)
            mu_base = b["sum_mu_base"] / c
            std_base = b["sum_std_base"] / c
            mu_rel = b["sum_mu_rel"] / c
            std_rel = b["sum_std_rel"] / c
            r = b["sum_r"] / c
            kl = b["sum_kl"] / c
            abs_base = b["sum_abs_base"] / c
            abs_rel = b["sum_abs_rel"] / c
            rel_over_eb = abs_rel / abs_base.clamp_min(self.eps)
            rec = {
                "count": float(c.mean().item()),
                "mu_base": float(mu_base.mean().item()),
                "std_base": float(std_base.mean().item()),
                "mu_rel": float(mu_rel.mean().item()),
                "std_rel": float(std_rel.mean().item()),
                "r": float(r.mean().item()),
                "kl": float(kl.mean().item()),
                "abs_base": float(abs_base.mean().item()),
                "abs_rel": float(abs_rel.mean().item()),
                "rel_over_eb": float(rel_over_eb.mean().item()),
                "std_base_q": self._quantile_dict(b.get("samples_std_base", [])),
                "std_rel_q": self._quantile_dict(b.get("samples_std_rel", [])),
                "r_q": self._quantile_dict(b.get("samples_r", [])),
                "kl_q": self._quantile_dict(b.get("samples_kl", [])),
            }
            if self.per_head:
                rec["mu_base_per_head"] = mu_base.detach().float().cpu().tolist()
                rec["std_base_per_head"] = std_base.detach().float().cpu().tolist()
                rec["mu_rel_per_head"] = mu_rel.detach().float().cpu().tolist()
                rec["std_rel_per_head"] = std_rel.detach().float().cpu().tolist()
                rec["r_per_head"] = r.detach().float().cpu().tolist()
                rec["kl_per_head"] = kl.detach().float().cpu().tolist()
                rec["abs_base_per_head"] = abs_base.detach().float().cpu().tolist()
                rec["abs_rel_per_head"] = abs_rel.detach().float().cpu().tolist()
                rec["rel_over_eb_per_head"] = rel_over_eb.detach().float().cpu().tolist()
            out[int(bidx)] = rec
        return out
def _make_router_mlp(hidden_size: int, out_dim: int, use_non_linear: bool) -> nn.Sequential:
    if use_non_linear:
        return nn.Sequential(
            nn.Linear(hidden_size, 32, bias=False),
            NewGELUActivation(),
            nn.Linear(32, out_dim, bias=False),
        )
    return nn.Sequential(
        nn.Linear(hidden_size, 32, bias=False),
        nn.Linear(32, out_dim, bias=False),
    )

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_block(save_dir: str, name: str, payload: dict):
    path = os.path.join(save_dir, name)
    torch.save(payload, path)

@torch.no_grad()
def dump_last_query_per_dim(
    save_path: str,
    q: torch.Tensor,                 # [B,T,H,D]
    k: torch.Tensor,                 # [B,T,H,D]
    w: torch.Tensor,                 # [B,T,H,D]
    M: torch.Tensor,                 # [B,H,T,T]
    wavelet_dtt: torch.Tensor | None = None,  # [D,T,T] or None
    layer_idx: int = 0,
    save_dtype: torch.dtype = torch.float16,
    compute_corr_row: bool = False,  # 是否直接算 corr_row_perd（会多一次计算）
    router1=None,
    router2=None,
):
    """
    Save per-d contributions for the last query i=T-1 against all keys n=0..T-1.

    Always saved (baseline-related):
      - lower_qk_row:   [B,H,T,D]
      - strict_wk_row:  [B,H,T,D]   (j=i row of strictLower(WK^T) per-d)
      - M_row:          [B,H,T]
      - corr_row_perd:  [B,H,T,D]   (optional; last-row of (M@strictLower(WK^T)) per-d)

    Saved only when wavelet_dtt is not None:
      - rel1_qP_row:    [B,H,T,D]
      - rel2_mwP_row:   [B,H,T,D]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B, T, H, D = q.shape
    assert k.shape == (B, T, H, D)
    assert w.shape == (B, T, H, D)
    assert M.shape == (B, H, T, T)
    last_router1 = router1[:, -1, :, :] if router1 is not None else None
    last_router2 = router2[:, -1, :, :] if router2 is not None else None # B H S
    if wavelet_dtt is not None:
        assert wavelet_dtt.shape == (D, T, T), (wavelet_dtt.shape, (D, T, T))

    i = T - 1  # last query

    # [B,H,T,D] layout for easier row extraction
    q_bhtd = q.permute(0, 2, 1, 3).contiguous()
    k_bhtd = k.permute(0, 2, 1, 3).contiguous()
    w_bhtd = w.permute(0, 2, 1, 3).contiguous()

    # last-row slices: q_i, w_i : [B,H,D]
    q_i = q_bhtd[:, :, i, :]          # [B,H,D]
    w_i = w_bhtd[:, :, i, :]          # [B,H,D]
    k_all = k_bhtd                    # [B,H,T,D]

    # ---- (C) lower_qk_row_perd: q_{i,d} * k_{n,d} ----
    # last row i=T-1 => lower mask allows all n, so no mask needed
    qk = torch.einsum("b h d, b h n d -> b h n d", q_i, k_all)        # [B,H,T,D]

    # ---- (D) strict_wk_row_perd: w_{j,d} * k_{n,d} with j=i ----
    # strictLower mask requires j>n, so for last row: mask n<=T-2 (and zero at n==T-1)
    wk = torch.einsum("b h d, b h n d -> b h n d", w_i, k_all)        # [B,H,T,D]
    n_mask = (torch.arange(T, device=q.device) < i)                   # n < T-1
    wk = wk.masked_fill(~n_mask[None, None, :, None], 0.0)

    # ---- optional: corr_row_perd = last-row of (M @ strictLower(WK^T)) per-d ----
    corr_row = None
    if compute_corr_row:
        # corr[i,n,d] = k[n,d] * sum_{j>n} M[i,j] * w[j,d]
        Mi = M[:, :, i, :]                               # [B,H,T]
        w_bthd = w.permute(0, 2, 1, 3).contiguous()       # [B,H,T,D]
        weighted_w = w_bthd * Mi.unsqueeze(-1)            # [B,H,T,D]

        # suffix sum: suf[n,d] = sum_{j>=n} weighted_w[j,d]
        suf = torch.flip(torch.cumsum(torch.flip(weighted_w, dims=[2]), dim=2), dims=[2])  # [B,H,T,D]
        # sum_{j>n} => suf[n+1]
        suf_shift = torch.zeros_like(suf)
        suf_shift[:, :, :-1, :] = suf[:, :, 1:, :]        # [B,H,T,D]
        corr_row = k_all * suf_shift                      # [B,H,T,D]

    # ---- (A)(B) wavelet rel terms, only if wavelet_dtt is provided ----
    rel1 = None
    rel2 = None
    if wavelet_dtt is not None:
        P_i = wavelet_dtt[:, i, :]  # [D,T]

        # rel1_qP_row_perd: q_{i,d} * P_{d,i,n}
        rel1 = torch.einsum("b h d, d n -> b h n d", q_i, P_i)  # [B,H,T,D]
        rel1 = rel1 * last_router1[:,:,None,:].repeat_interleave(8, dim=-1)
        # rel2_mwP_row_perd: (MW)_{i,d} * P_{d,i,n}
        MW_i = torch.einsum("b h j, b j h d -> b h d", M[:, :, i, :], w)  # w: [B,T,H,D]
        rel2 = torch.einsum("b h d, d n -> b h n d", MW_i, P_i)           # [B,H,T,D]
        rel2 = rel2 * last_router2[:,:,None,:].repeat_interleave(8, dim=-1)
    payload = {
        "layer_idx": layer_idx,
        "i": i,
        "B": B, "T": T, "H": H, "D": D,
        "lower_qk_row": qk.to(save_dtype).cpu(),
        "strict_wk_row": wk.to(save_dtype).cpu(),
        "M_row": M[:, :, i, :].detach().to(save_dtype).cpu(),  # [B,H,T]
        "has_wavelet": (wavelet_dtt is not None),
    }
    if rel1 is not None and rel2 is not None:
        payload["rel1_qP_row"] = rel1.to(save_dtype).cpu()
        payload["rel2_mwP_row"] = rel2.to(save_dtype).cpu()
    if corr_row is not None:
        payload["corr_row_perd"] = corr_row.to(save_dtype).cpu()

    torch.save(payload, save_path)
    print(f"[done] saved last-query per-d terms to: {save_path}")

def sample_index_pairs(
    block_size: int,
    num_samples: int,
    *,
    deltas: torch.Tensor | None = None,
    min_delta: int = 1,
    max_delta: int | None = None,
    method: str = "mix",           # "mix" | "uniform" | "geometric"
    geom_p: float = 0.2,           # 几何分布参数（期望 ~ 1/p），仅当 method in {"mix","geometric"} 时使用
    uniform_frac: float = 0.3,     # mix 模式下，均匀采样比例
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    allow_delta_zero: bool = False # 如需 Δ=0（自指）则设 True
):
    """
    随机采样 (i, j) 索引对，满足 j = i + Δ，且 0 <= i < j < block_size（若 allow_delta_zero=True 则允许 i==j）。
    - 若提供 deltas，则按给定 Δ 向量逐一采样 (i, j)；
    - 否则按 method 生成长度为 num_samples 的 Δ 向量。

    返回:
        i_idx: LongTensor[num_samples]
        j_idx: LongTensor[num_samples]
        deltas: LongTensor[num_samples]  # 实际使用的 Δ
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if max_delta is None:
        max_delta = block_size - 1

    # 边界处理
    if allow_delta_zero:
        min_delta = 0
    else:
        min_delta = max(1, min_delta)
    max_delta = max(min_delta, min(max_delta, block_size - (0 if allow_delta_zero else 1)))

    if max_delta < min_delta:
        raise ValueError(f"无可用的 Δ 范围：min_delta={min_delta}, max_delta={max_delta}, block_size={block_size}")

    # 1) 生成/规范化 Δ
    if deltas is None:
        if method == "uniform":
            deltas = torch.randint(
                low=min_delta,
                high=max_delta + 1,
                size=(num_samples,),
                device=device,
                generator=generator,
            )
        elif method == "geometric":
            # 采几何分布，截断到 [min_delta, max_delta]
            # PyTorch 没有直接的几何分布，这里用伯努利和求首个成功的思路不高效；
            # 用近似：先按几何的 PMF 构造离散表，再从表中采样（高效且可向量化）
            support = torch.arange(min_delta, max_delta + 1, device=device)
            # 几何分布（从1开始）的PMF: p*(1-p)^(k-1)，这里平移到 min_delta 起点
            shifted = support - (1 if not allow_delta_zero else 0)
            pmf = (geom_p * torch.pow(1 - geom_p, shifted - 1)).clamp_min(1e-12)
            pmf = pmf / pmf.sum()
            # 多项式采样
            deltas = support[torch.multinomial(pmf, num_samples, replacement=True, generator=generator)]
        elif method == "mix":
            # 按 uniform_frac 混合均匀与几何
            num_u = int(num_samples * uniform_frac)
            num_g = num_samples - num_u
            deltas_u = torch.randint(
                low=min_delta, high=max_delta + 1, size=(num_u,), device=device, generator=generator
            )
            support = torch.arange(min_delta, max_delta + 1, device=device)
            shifted = support - (1 if not allow_delta_zero else 0)
            pmf = (geom_p * torch.pow(1 - geom_p, shifted - 1)).clamp_min(1e-12)
            pmf = pmf / pmf.sum()
            deltas_g = support[torch.multinomial(pmf, num_g, replacement=True, generator=generator)]
            deltas = torch.empty(num_samples, device=device, dtype=torch.long)
            deltas[:num_u] = deltas_u
            deltas[num_u:] = deltas_g
            # 打乱
            perm = torch.randperm(num_samples, device=device, generator=generator)
            deltas = deltas[perm]
        else:
            raise ValueError(f"未知 method: {method}")
    else:
        deltas = deltas.to(device=device, dtype=torch.long)
        if (deltas < min_delta).any() or (deltas > max_delta).any():
            raise ValueError(f"deltas 超出范围 [{min_delta}, {max_delta}]")

        if deltas.numel() != num_samples:
            # 若用户给的 deltas 数量与 num_samples 不一致，则循环扩展或截断
            reps = (num_samples + deltas.numel() - 1) // deltas.numel()
            deltas = deltas.repeat(reps)[:num_samples]

    # 2) 对每个 Δ 采样左索引 i，使得 i ∈ [0, block_size - 1 - Δ]（包含），然后 j = i + Δ
    # 先计算各样本对应的上界（含）
    # 有的 Δ 可能接近 block_size-1，此时合法 i 的选择很少；下面逐样本向量化处理。
    # 构造每个样本的 i_max = block_size - 1 - Δ
    i_max = (block_size - 1) - deltas
    # 对应的“可采样长度” = i_max + 1，最小为 1（保证至少一个位置）
    span = i_max + 1
    # 为了一次性采样，先对每个样本生成一个 [0, span_s) 的随机数，再拼成 i
    # 方案：先采 [0, 1) 浮点，再乘以 span，取 floor
    # 但要保证 span>0，这里根据构造一定成立
    rand_u = torch.rand(num_samples, device=device, generator=generator)
    i_idx = (rand_u * span.to(rand_u.dtype)).floor().to(torch.long)
    # i 的真实值 = i_idx（已是合法范围内）
    j_idx = i_idx + deltas

    # 3) 安全断言（可选）
    # (i_idx >= 0).all(), (j_idx < block_size).all()

    return i_idx, j_idx, deltas
def build_causal_mask(q_len: int,
                      k_len: int | None = None,
                      past_kv_len: int = 0,
                      device=None,
                      dtype=torch.bool):
    """
    返回形状 (1, 1, q_len, k_len_total) 的布尔mask，
    True=保留；False=屏蔽。
    k_len_total = (past_kv_len + (k_len if k_len is not None else q_len))
    """
    if k_len is None:
        k_len = q_len
    total_k = past_kv_len + k_len
    i = torch.arange(q_len, device=device).unsqueeze(1) + past_kv_len  # query 位置（对齐到全局K坐标）
    j = torch.arange(total_k, device=device).unsqueeze(0)               # key 位置（全局）
    mask = (i >= j)  # 下三角（含对角）
    return mask.view(1, 1, q_len, total_k).to(dtype)
def log_line_fig(tensor, name, y_min=None, y_max=None, sigma=200):    
    # 将 tensor 转换为 numpy 数组
    tensor = tensor.cpu().detach().numpy()

    # 对 tensor 进行高斯平滑
    smoothed_tensor = gaussian_filter1d(tensor, sigma=sigma)

    # 找到原始 tensor 中的 peaks
    peaks, _ = find_peaks(tensor, distance=200)

    # 绘制原始折线图和高斯平滑曲线
    plt.figure(figsize=(10, 6))
    plt.plot(tensor, label='Tensor Values')
    plt.plot(smoothed_tensor, color='red', label='Smoothed Tensor (Gaussian)')  # 添加高斯平滑曲线
    plt.scatter(peaks, tensor[peaks], color='green', label='Peaks')  # 峰值标记为绿色

    # 在峰值处添加文本标注
    for peak in peaks:  
        plt.text(peak, tensor[peak], f'{peak}', fontsize=9, ha='right')

    # 设置 y 轴范围
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    
    # 添加标签和标题
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Plot of 1D Tensor with Gaussian Smoothing and Peaks Highlighted')
    plt.legend(['Tensor Values', 'Smoothed Tensor (Gaussian)', 'Peaks'])

    # 保存图像
    plt.savefig(f'{name}.png')

    # 将图像上传到wandb（此处注释了）
    # wandb.log({"line_plot": wandb.Image(f'line_plot_{name}.png')})

    plt.close()

def log_heatmap(tensor, name = '', vmin=-1, vmax = 0):
    
    # 使用 seaborn 生成 heatmap
    plt.imshow(tensor, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()  # 添加颜色条以显示数值范围
    plt.savefig(f'heatmap.png')    
    plt.close()

    # 记录到 wandb
    # wandb.log({f"heatmap_{name}": wandb.Image(f"heatmap.png")})



def compute_wavelet_scores_batched(
    q: torch.Tensor,                  # [B, H, D]
    k: torch.Tensor,                  # [B, T, H, D]
    wavelet_decay_list: torch.Tensor, # [D, L]  相对距离表
    *,
    table_direction: str = "near_to_far",  # or "far_to_near"
) -> torch.Tensor:

    _, _, D = q.shape
    assert wavelet_decay_list.dim()==2 and wavelet_decay_list.size(0)==D, "wavelet_decay_list 应为 [D,L]"

    device = q.device
    dtype  = q.dtype
    L = wavelet_decay_list.size(1)
    scores = q[:, None, :, :] * k # [S,1,H,D] * [S,T,H,D] -> [S,T,H,D]
    scores = scores * wavelet_decay_list.transpose(0, 1)[None, :, None, :] # [S,T,H,D] * [1,T,1,D] -> [S,T,H,D]
    return scores
# def compute_wavelet_score_single(q_j, k_i, wavelet_decay_list, i_idx, j_idx, *, sqrt_d_scale=True):
#     H, D = q_j.shape
#     rel = i_idx - j_idx                     # 假设 0<=rel<R
#     d = wavelet_decay_list[:, rel].to(q_j)     # [D]
#     q_w = q_j * d                              # [H,D]
#     score = torch.einsum('hd,hd->', q_w, k_i)  # 标量

#     return score / math.sqrt(D)
def compute_path_scores_batched_last_q(
    q: torch.Tensor,          # [B, H, D]   —— 使用最后一个 query（i = T-1）
    k: torch.Tensor,          # [B, T, H, D]
    w: torch.Tensor,          # [B, T, H, D]
    beta: torch.Tensor,       # [B, T, H]
    *,
    sqrt_d_scale: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    计算 Path Attention 下，最后一个 query（i = T-1）与所有 key(j) 的逐维匹配分数：
        scores[b, j, h, d] = q[b, h, d] * x_j[b, h, d] / sqrt(D),
    其中
        x_j = ( ∏_{t=j}^{T-2} (I - β_{b,t,h} w_{b,t,h} w_{b,t,h}^T) ) k[b, j, h, :]
    注意：当 j = T-1 时，空积为 I，故 x_{T-1} = k[b, T-1, h, :]

    返回: scores [B, T, H, D]（不沿 D 聚合，留给上层在频域做处理）
    复杂度: O(B·H·D·T²)
    """

    # 形状与设备检查
    assert q.dim() == 3,          "q 应为 [B,H,D]"
    assert k.dim() == 4,          "k 应为 [B,T,H,D]"
    assert w.dim() == 4,          "w 应为 [B,T,H,D]"
    assert beta.dim() == 3,       "beta 应为 [B,T,H]"

    B, H, D = q.shape
    assert k.size(0) == B and k.size(2) == H and k.size(3) == D, "k 维度与 q 不匹配"
    assert w.size(0) == B and w.size(2) == H and w.size(3) == D, "w 维度与 q 不匹配"
    assert beta.size(0) == B and beta.size(2) == H, "beta 维度与 q 不匹配"

    T = k.size(1)
    assert w.size(1) == T and beta.size(1) == T, "w/beta 的 T 维需与 k 一致"

    device = q.device
    dtype  = q.dtype

    q = q.to(device=device, dtype=dtype)
    k = k.to(device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)
    beta = beta.to(device=device, dtype=dtype)

    # 结果张量
    scores = torch.empty((B, T, H, D), device=device, dtype=dtype)

    # 预取最后一个 query
    # 形状对齐方便广播: [B,1,H,D]
    q_last = q[:, None, :, :]   # [B,1,H,D]

    j_iter = range(T)

    # 对每个 j：应用从 t=j 到 t=T-2 的 rank-1 线性变换
    # x <- x - beta_t * <x, w_t> * w_t
    for j in j_iter:
        # 初始向量：每个 batch、每个 head 的 k_j
        x = k[:, j, :, :].clone()         # [B,H,D]

        # 空积情形：j==T-1 不需要任何变换
        if j < T - 1:
            # 逐 t 累乘算子（对所有 batch/head 同时进行，向量化），成本 O(B·H·D·(T-j))
            for t in range(j, T - 1):
                w_t = w[:, t, :, :]                   # [B,H,D]
                b_t = beta[:, t, :].unsqueeze(-1)     # [B,H,1]
                # 逐 head 点积：<x, w_t> -> [B,H,1]
                dot = (x * w_t).sum(dim=-1, keepdim=True)  # [B,H,1]
                # 应用 rank-1 更新
                x = x - b_t * dot * w_t               # [B,H,D]

        # 与最后 query 做逐维匹配（不沿 D 求和，留频域）
        scores[:, j, :, :] = q_last[:, 0, :, :] * x   # [B,H,D]

    return scores
import random
import torch
def make_highfreq_weight(K, alpha=1.0, device="cpu"):
    # 频率 index: 0 ~ K-1
    k = torch.arange(K, device=device, dtype=torch.float32)
    # 归一化到 [0,1]
    f = k / (K - 1)
    # 高频权重大一点，比如 f^alpha
    w = f**alpha
    # 也可以加一个 floor，避免前面直接变 0
    # w = (f**alpha) + 0.1
    return w  # [K]
def spectrum_over_T_multi(x: torch.Tensor, eps: float = 1e-6):
    """
    x: [B, Q, T, H, D]
    沿 T 维 (dim=2) 做 rFFT，得到每个 (B,Q,H,D) 上的频谱。
    
    返回:
        A    : [B, Q, K, H, D]   幅值
        A_log: [B, Q, K, H, D]   log 幅值
        其中 K = T//2 + 1
    """
    assert x.dim() == 5, f"x 应为 [B, Q, T, H, D]，当前 {x.shape}"
    X = torch.fft.rfft(x, dim=2, norm='ortho')   # [B, Q, K, H, D]
    A = X.abs()
    A_log = torch.log(A.clamp_min(eps))
    return A, A_log
    # return A

def spectral_distill_over_L_cos(
    student: torch.Tensor,   # [B, L, H, D] or [B, Q, T, H, D]
    teacher: torch.Tensor,   # [B, L, H, D] or [B, Q, T, H, D]
    *,
    w=1.0,
    eps: float = 1e-8,
):
    """
    频谱 Shape 蒸馏（COSINE VERSION）
    ----------------------------------------------------
    1) 沿 T 做 rFFT → amplitude A_s, A_t: [B, Q, K, H, D]
    2) 在频率 K 维上做 cosine similarity
    3) loss = (1 - cosine) 的均值
    """

    # Support both shapes: [B,L,H,D] → [B,1,T,H,D]
    if student.dim() == 4:
        x_s = student.unsqueeze(1)
        x_t = teacher.unsqueeze(1)
    elif student.dim() == 5:
        x_s = student
        x_t = teacher
    else:
        raise ValueError(f"Bad shape: {student.shape}")

    # --------------------------------------------------
    # 1) rFFT amplitude spectrum: [B, Q, K, H, D]
    # --------------------------------------------------
    with torch.no_grad():
        A_t, _ = spectrum_over_T_multi(x_t)       # teacher 不反传
    A_s, _ = spectrum_over_T_multi(x_s)

    # --------------------------------------------------
    # reshape 到 [B*Q*H*D, K]，方便批量计算 cosine
    # --------------------------------------------------
    B, Q, K, H, D = A_s.shape
    A_t_flat = A_t.permute(0,1,3,4,2).reshape(-1, K)
    A_s_flat = A_s.permute(0,1,3,4,2).reshape(-1, K)

    # --------------------------------------------------
    # 2) 计算 cosine similarity
    # --------------------------------------------------
    dot = (A_t_flat * A_s_flat).sum(dim=-1)                     # [N]
    norm_t = A_t_flat.norm(dim=-1)
    norm_s = A_s_flat.norm(dim=-1)

    cosine = dot / (norm_t * norm_s + eps)                      # [N]

    # --------------------------------------------------
    # 3) loss = mean(1 - cosine)
    # --------------------------------------------------
    loss_cos = (1.0 - cosine).mean() * w

    return loss_cos
def spectral_distill_over_L_mse(
    student: torch.Tensor,   # [B, L, H, D]  (path_attn_scores 映射/reshape到该形状)
    teacher: torch.Tensor,   # [B, L, H, D]  (rotary 或 wavelet 的 logits 映射/reshape)
    *,
    w=1.0,
    tau: float = 1.0,                      # 暂时不用，保留接口
    lambda_mse: float = 1.0,               # 形状 MSE 权重
    lambda_kl: float = 0.5,                # KL 权重
    lambda_cos: float = 0.0,               # 暂不使用
    eps: float = 1e-8,
):
    """
    频谱“形状”蒸馏（只对齐频率方向分布，不对齐绝对幅值）：

      1. 对 teacher / student 沿 L 维做 rFFT，得到幅值 A_t, A_s: [B, Q=1, K, H, D]
      2. 在 K 维上对幅值做归一化：
            p_t = A_t / sum_K A_t
            p_s = A_s / sum_K A_s
         视作“在频率上的分布”
      3. loss = lambda_kl * KL(p_t || p_s) + lambda_mse * MSE(log p)

    这样模型不能靠整体缩小 amplitude 来“逃避”蒸馏，只能去贴频谱 shape，
    同时不会像原来的 log-MSE 那样强迫绝对值完全对齐。
    """
    # 兼容 [B, L, H, D] 或 [B, Q, T, H, D]
    if student.dim() == 4:
        # [B, L, H, D] -> [B, Q=1, T=L, H, D]
        x_s = student.unsqueeze(1)
        x_t = teacher.unsqueeze(1)
    elif student.dim() == 5:
        x_s = student
        x_t = teacher
    
    else:
        raise ValueError(f"student 形状必须是 [B,L,H,D] 或 [B,Q,T,H,D]，当前 {student.shape}")

    # 1) rFFT 得到幅值谱: [B, Q, K, H, D]
    with torch.no_grad():
        A_t, A_t_log = spectrum_over_T_multi(x_t)   # teacher 不反传
    A_s, A_s_log = spectrum_over_T_multi(x_s)

    loss_spec_mse = ((A_t_log - A_s_log) ** 2 * w).mean()
    return loss_spec_mse
    loss = lambda_kl * loss_kl + lambda_mse * loss_mse

    return loss

def path_attn_last_query_elementwise(Q_last, K, W, beta):
    """
    使用 UT 分解计算 PaTH 的“有效 q 向量”，只针对最后一个 query token。
    
    参数:
        Q_last : [B, 1, H, D]
            已经是「最后一个 token」的 query，通常来自：
                Q_last = Q_all[:, -1:, :, :]
        K      : [B, T, H, D]
        W      : [B, T, H, D]    PaTH 中的 Householder 向量 w_t
        beta   : [B, T, H]       Householder 系数 β_t

    返回:
        out    : [B, T, H, D]
            out[b, t, h, d] = K[b, t, h, d] * (Q_last[b, 0, h, d] - S_{b,h,t+1,d})
            其中 S_{b,h,t+1} 是根据 UT 分解得到的修正向量。
            对最后一位 t = T-1，S_{b,h,T} 视为 0。
    """
    B, one, H, D = Q_last.shape
    assert one == 1, "Q_last 应为 [B, 1, H, D]，且只包含最后一个 token"
    assert K.shape == (B, K.shape[1], H, D)
    assert W.shape == K.shape
    assert beta.shape == (B, K.shape[1], H)

    B_, T, H_, D_ = K.shape
    assert B_ == B and H_ == H and D_ == D

    device = Q_last.device
    dtype = Q_last.dtype

    # 把 (B, H) 拉平成一个大的 batch 维 BH，方便用 batched 矩阵运算
    BH = B * H

    # Q_last: [B, 1, H, D] -> [BH, D]
    # 注意：这里的这个 token 就是“原序列的最后一个 token”
    q_last = Q_last.reshape(B, 1, H, D)[:, 0, :, :].reshape(BH, D)  # [BH, D]

    # K_flat, W_flat: [B, T, H, D] -> [BH, T, D]
    K_flat = K.permute(0, 2, 1, 3).reshape(BH, T, D)      # [BH, T, D]
    W_flat = W.permute(0, 2, 1, 3).reshape(BH, T, D)      # [BH, T, D]

    # beta_flat: [B, T, H] -> [BH, T]
    beta_flat = beta.permute(0, 2, 1).reshape(BH, T)      # [BH, T]

    # ===== 1) 构造 batched 的 WDWT = (W * beta) @ W^T =====
    B_scaled = W_flat * beta_flat.unsqueeze(-1)           # [BH, T, D]
    WDWT = B_scaled @ W_flat.transpose(1, 2)              # [BH, T, T]

    # strict lower 部分 + I -> T_basic = I + strictLower(WDW^T)
    Lmat = torch.tril(WDWT, diagonal=-1)                  # [BH, T, T]
    I = torch.eye(T, device=device, dtype=dtype).expand(BH, T, T)
    T_basic = I + Lmat                                    # [BH, T, T] 下三角

    # ===== 2) 计算 x = T_basic^{-1} (beta ⊙ (W q_last)) =====
    # t = W q_last : [BH, T]，用逐元素乘再 sum 避免多一次 @
    t = (W_flat * q_last.unsqueeze(1)).sum(dim=-1)        # [BH, T]
    z = beta_flat * t                                     # [BH, T]

    x = torch.linalg.solve_triangular(
        T_basic, z.unsqueeze(-1), upper=False
    ).squeeze(-1)                                         # [BH, T]

    # ===== 3) U_m = x_m * w_m, 做 suffix sum 得到 S_t =====
    U = W_flat * x.unsqueeze(-1)                          # [BH, T, D]

    # S[t] = sum_{m=t}^{T-1} U[m]
    U_flip = torch.flip(U, dims=[1])                      # 反转时间维
    S_flip = torch.cumsum(U_flip, dim=1)
    S = torch.flip(S_flip, dims=[1])                      # [BH, T, D]

    # S_shift[j] = S[j+1]，最后一个位置的修正为 0
    S_shift = torch.zeros_like(S)
    if T > 1:
        S_shift[:, :-1] = S[:, 1:]                        # S_shift[:, j] = S[:, j+1]

    # ===== 4) 有效 q: q_eff[j] = q_last - S_{j+1} =====
    q_eff = q_last.unsqueeze(1) - S_shift                 # [BH, T, D]

    # 按维度乘以 key，得到你要的 [BH, T, D]
    out_flat = K_flat * q_eff                             # [BH, T, D]

    # reshape 回 [B, T, H, D]
    out = out_flat.reshape(B, H, T, D).permute(0, 2, 1, 3)

    return out
def path_attn_multi_query_elementwise(Q_sel, K, W, beta, offsets=(1, 8, 16, 32)):
    """
    使用 UT 分解，针对多个 query 位置一次性计算 PaTH 的“有效 q 向量”，
    并输出与所有 K 的逐维乘积（带严格 causal：未来 reflectors 不影响当前 query）。

    参数:
        Q_sel  : [B, Q, H, D]，这里 Q = 4
                 第 q 个 query 是从原序列末尾数 offsets[q] 个位置：
                     idx_q = T - offsets[q]
        K      : [B, T, H, D]
        W      : [B, T, H, D]
        beta   : [B, T, H]
        offsets: 长度 Q 的 tuple，例如 (1, 8, 16, 32)，表示相对末尾的偏移

    返回:
        out    : [B, Q, T, H, D]
                 out[b, q, j, h, d]
                 = 1_{j <= i_q} * K[b, j, h, d] * ( q_{b,q,h,d} - S^{(b,h,q)}_{j+1,d} )
    """
    B, Q, H, D = Q_sel.shape
    assert Q == len(offsets), "Q_sel 第二维和 offsets 长度必须一致"
    assert K.shape == (B, K.shape[1], H, D)
    assert W.shape == K.shape
    assert beta.shape == (B, K.shape[1], H)

    B_, T, H_, D_ = K.shape
    assert B_ == B and H_ == H and D_ == D

    device = Q_sel.device
    dtype = Q_sel.dtype
    BH = B * H   # 把 (B,H) 合并成大 batch 维

    # ---- 0) 预处理：把 (B,H) 拉平 ----
    # Q_sel: [B, Q, H, D] -> [BH, Q, D]
    q_sel = Q_sel.permute(0, 2, 1, 3).reshape(BH, Q, D)          # [BH, Q, D]

    # K, W: [B, T, H, D] -> [BH, T, D]
    K_flat = K.permute(0, 2, 1, 3).reshape(BH, T, D)             # [BH, T, D]
    W_flat = W.permute(0, 2, 1, 3).reshape(BH, T, D)             # [BH, T, D]

    # beta: [B, T, H] -> [BH, T]
    beta_flat = beta.permute(0, 2, 1).reshape(BH, T)             # [BH, T]

    # ---- 1) 构造 T_basic = I + strictLower(W D W^T) ----
    B_scaled = W_flat * beta_flat.unsqueeze(-1)                  # [BH, T, D]
    WDWT = B_scaled @ W_flat.transpose(1, 2)                     # [BH, T, T]

    Lmat = torch.tril(WDWT, diagonal=-1)                         # strict lower
    I = torch.eye(T, device=device, dtype=dtype).expand(BH, T, T)
    T_basic = I + Lmat                                           # [BH, T, T] 下三角

    # ---- 2) 为每个 query 位置 i_q 构造右侧 mask M_R^{(i_q)} ----
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)  # [Q]
    pos = T - offsets                                            # [Q], 绝对下标 i_q
    pos = torch.clamp(pos, 0, T - 1)

    t_idx = torch.arange(T, device=device)                       # [T]
    # 对 reflector 索引 m: mask_R[q, m] = 1_{ m <= i_q }
    mask_R = (t_idx.unsqueeze(0) <= pos.unsqueeze(1))            # [Q, T]
    mask_R = mask_R.to(dtype)                                    # float 型

    # ---- 3) 计算 y^{(q)} = T^{-1} (W⊙M_R^{(i_q)}) q^{(q)} ----
    # t_all[b,h,q,t] = w_t^T q^{(q)}
    t_all = (W_flat.unsqueeze(1) * q_sel.unsqueeze(2)).sum(dim=-1)    # [BH, Q, T]

    # z = beta ⊙ mask_R ⊙ (W q)
    z = beta_flat.unsqueeze(1) * t_all * mask_R.unsqueeze(0)          # [BH, Q, T]

    x = torch.linalg.solve_triangular(
        T_basic.unsqueeze(1),    # [BH, 1, T, T]
        z.unsqueeze(-1),         # [BH, Q, T, 1]
        upper=False
    ).squeeze(-1)                                                    # [BH, Q, T]

    # 🔴 关键修复：显式清零 m > i_q 的系数，防止未来 reflectors 泄漏进来
    mask_R_bh = mask_R.unsqueeze(0)                                  # [1, Q, T]
    x = x * mask_R_bh                                                # [BH, Q, T]

    # ---- 4) U_m^{(q)} = x_m^{(q)} w_m，suffix sum 得到 S_t^{(q)} ----
    U = W_flat.unsqueeze(1) * x.unsqueeze(-1)                        # [BH, Q, T, D]

    U_flip = torch.flip(U, dims=[2])                                 # [BH, Q, T, D]
    S_flip = torch.cumsum(U_flip, dim=2)
    S = torch.flip(S_flip, dims=[2])                                 # [BH, Q, T, D]

    S_shift = torch.zeros_like(S)
    if T > 1:
        S_shift[:, :, :-1, :] = S[:, :, 1:, :]                       # [BH, Q, T, D]

    # ---- 5) q_eff^{(q)}[j] = q^{(q)} - S^{(q)}_{j+1} ----
    q_eff = q_sel.unsqueeze(2) - S_shift                             # [BH, Q, T, D]

    # ---- 6) 与 K 做逐维乘积 ----
    out_flat = K_flat.unsqueeze(1) * q_eff                           # [BH, Q, T, D]

    # ---- 7) key 维度的 causal mask：对 j > i_q 置 0（可选，但建议保留）----
    mask_key = (t_idx.unsqueeze(0) <= pos.unsqueeze(1)).to(dtype)    # [Q, T]
    mask_key = mask_key.unsqueeze(0).unsqueeze(-1)                   # [1, Q, T, 1]
    out_flat = out_flat * mask_key                                   # [BH, Q, T, D]

    # reshape 回 [B, Q, T, H, D]
    out = out_flat.reshape(B, H, Q, T, D).permute(0, 2, 3, 1, 4)

    return out
def _cosine_band(a_s: torch.Tensor,
                 a_t: torch.Tensor,
                 eps: float = 1e-8,
                 zero_mean: bool = True) -> torch.Tensor:
    """
    a_s, a_t: [N, K_band]
    返回该频带上的 cos，相当于相关系数（若 zero_mean=True）
    """
    if zero_mean:
        a_s = a_s - a_s.mean(dim=-1, keepdim=True)
        a_t = a_t - a_t.mean(dim=-1, keepdim=True)

    dot = (a_s * a_t).sum(dim=-1)               # [N]
    norm_s = a_s.norm(dim=-1)
    norm_t = a_t.norm(dim=-1)

    cosine = dot / (norm_s * norm_t + eps)      # [N]
    return cosine


def spectral_distill_over_L_cos_3bands(
    student: torch.Tensor,   # [B, L, H, D] or [B, Q, T, H, D]
    teacher: torch.Tensor,   # [B, L, H, D] or [B, Q, T, H, D]
    *,
    w: float = 1.0,
    eps: float = 1e-8,
    zero_mean: bool = True,     # 建议先开着，更接近 shape 对齐
):
    """
    频谱 Shape 蒸馏（三段 COSINE 版本）
    ----------------------------------------------------
    1) 沿 T 做 rFFT → A_s, A_t: [B, Q, K, H, D]
    2) 在频率轴 K 上分成 3 段，分别算 cosine：
         - low  : [0, k1)
         - mid  : [k1, k2)
         - high : [k2, K)
    3) loss = 各段 (1 - cosine) 的平均
    """

    # 支持 [B,L,H,D] 或 [B,Q,T,H,D]
    if student.dim() == 4:
        x_s = student.unsqueeze(1)   # [B,1,L,H,D]
        x_t = teacher.unsqueeze(1)
    elif student.dim() == 5:
        x_s = student
        x_t = teacher
    else:
        raise ValueError(f"Bad shape: {student.shape}")

    # 1) rFFT → amplitude: [B, Q, K, H, D]
    with torch.no_grad():
        A_t, _ = spectrum_over_T_multi(x_t)   # teacher 不反传
    A_s, _ = spectrum_over_T_multi(x_s)

    B, Q, K, H, D = A_s.shape

    # reshape → [N, K]
    A_t_flat = A_t.permute(0, 1, 3, 4, 2).reshape(-1, K)  # [N, K]
    A_s_flat = A_s.permute(0, 1, 3, 4, 2).reshape(-1, K)  # [N, K]

    # 按 K 维分三段
    seg = K // 3
    k1 = seg
    k2 = 2 * seg
    # 三段：0:k1, k1:k2, k2:K（最后一段自动吃掉 remainder）
    bands = [
        (0, k1),      # low
        (k1, k2),     # mid
        (k2, K),      # high
    ]

    cos_list = []
    for start, end in bands:
        if end - start <= 1:  # 太短就跳过
            continue
        s_band = A_s_flat[:, start:end]
        t_band = A_t_flat[:, start:end]
        cos_band = _cosine_band(s_band, t_band, eps=eps, zero_mean=zero_mean)
        cos_list.append(cos_band)

    if not cos_list:
        # 异常情况：K 太小
        return A_s_flat.new_tensor(0.0)

    # 拼起来 → [num_bands * N]
    cosine_all = torch.cat(cos_list, dim=0)

    # loss = 各段 (1 - cos) 平均
    loss_cos = (1.0 - cosine_all).mean() * w
    return loss_cos
def compute_wavelet_scores_multi_causal(
    q: torch.Tensor,                  # [B, Q, H, D]
    k: torch.Tensor,                  # [B, T, H, D]
    wavelet_decay_list: torch.Tensor, # [D, L]  相对距离表 (delta >= 0)
    query_indices,                    # 长度 Q，可迭代，存放每个 query 的绝对位置 i_q
    *,
    table_direction: str = "near_to_far",  # or "far_to_near"
) -> torch.Tensor:
    """
    多个 query 的 wavelet PE teacher 版本（因果：只看左边）。
    对于第 q 个 query:
        - 位置 i_q = query_indices[q]
        - 只对 j <= i_q 的 key 生效
        - 相对距离 delta = i_q - j (>=0)，用 wavelet_decay_list[:, delta]

    参数:
        q  : [B, Q, H, D]
        k  : [B, T, H, D]
        wavelet_decay_list: [D, L]
        query_indices     : 长度 Q 的 list/1D tensor，元素在 [0, T-1]
        table_direction   : wavelet 表列方向

    返回:
        scores: [B, Q, T, H, D]
    """
    assert q.dim() == 4, "q 应为 [B, Q, H, D]"
    B, Q, H, D = q.shape
    assert k.shape[0] == B and k.shape[2] == H and k.shape[3] == D, "k 维度不匹配"
    _, T, _, _ = k.shape

    assert wavelet_decay_list.dim() == 2 and wavelet_decay_list.size(0) == D, \
        "wavelet_decay_list 应为 [D, L]"
    L = wavelet_decay_list.size(1)

    device = q.device
    dtype  = q.dtype

    # 1) 处理 query 的绝对位置 i_q
    query_indices = torch.as_tensor(query_indices, device=device, dtype=torch.long)  # [Q]
    assert query_indices.numel() == Q, "query_indices 长度必须和 Q 相同"

    # 2) 计算每个 (q, j) 的相对距离 delta = i_q - j
    t_idx = torch.arange(T, device=device)              # [T]
    # rel[q, j] = i_q - j
    rel = query_indices.unsqueeze(1) - t_idx.unsqueeze(0)  # [Q, T]

    # 合法区间：0 <= delta < L  (delta < 0 是右侧，将被 mask 掉)
    valid = (rel >= 0) & (rel < L)                      # [Q, T]
    rel_clamped = rel.clamp(0, L - 1)                   # [Q, T]

    # 3) 准备 wavelet 表（处理 near_to_far / far_to_near）
    decay_table = wavelet_decay_list.to(device=device, dtype=dtype)  # [D, L]
    if table_direction == "near_to_far":
        # 列 0 是 delta=0，列 1 是 delta=1，...
        base_table = decay_table.transpose(0, 1)        # [L, D]
    elif table_direction == "far_to_near":
        # 列 0 是最远的，flip 一下再当 near_to_far 用
        base_table = torch.flip(decay_table, dims=[1]).transpose(0, 1)  # [L, D]
    else:
        raise ValueError(f"未知 table_direction: {table_direction}")

    # 4) 为每个 (q, j) 按 delta 取出对应的 D 维衰减
    # decay_qt_d[q, j, d] = base_table[rel_clamped[q, j], d]
    decay_qt_d = base_table[rel_clamped]                # [Q, T, D]
    # 把右侧 (delta<0) 或超出 L 的位置置零
    decay_qt_d = decay_qt_d * valid.unsqueeze(-1)       # [Q, T, D]

    # 5) 扩展成 [1, Q, T, 1, D] 以便和 (B,Q,T,H,D) 广播
    decay = decay_qt_d.unsqueeze(0).unsqueeze(3)        # [1, Q, T, 1, D]

    # 6) 主体计算：q * k * decay
    # q: [B,Q,H,D] -> [B,Q,1,H,D]
    # k: [B,T,H,D] -> [B,1,T,H,D]
    scores = q[:, :, None, :, :] * k[:, None, :, :, :]  # [B,Q,T,H,D]
    scores = scores * decay                             # [B,Q,T,H,D]

    return scores
def sample_index_pairs(
    block_size: int,
    num_samples: int,
    *,
    deltas: torch.Tensor | None = None,
    min_delta: int = 1,
    max_delta: int | None = None,
    method: str = "uniform",           # "mix" | "uniform" | "geometric"
    geom_p: float = 0.2,           # 几何分布参数（期望 ~ 1/p），仅当 method in {"mix","geometric"} 时使用
    uniform_frac: float = 0.3,     # mix 模式下，均匀采样比例
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    allow_delta_zero: bool = False, # 如需 Δ=0（自指）则设 True
    i_bias_power: float = 3.0,      # 针对 i_idx 的左侧偏置系数，>1 时越偏左，=1 时均匀
):
    """
    随机采样 (i, j) 索引对，满足 j = i + Δ，且 0 <= i < j < block_size（若 allow_delta_zero=True 则允许 i==j）。

    - 若提供 deltas，则按给定 Δ 向量逐一采样 (i, j)；
    - 否则按 method 生成长度为 num_samples 的 Δ 向量。
    - i 的采样默认在 [0, block_size-1-Δ] 范围内均匀；
      若 i_bias_power > 1，则在该范围内对 i 进行“左侧偏置”，越大越偏向 0。

    返回:
        i_idx: LongTensor[num_samples]
        j_idx: LongTensor[num_samples]
        deltas: LongTensor[num_samples]  # 实际使用的 Δ
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if max_delta is None:
        max_delta = block_size - 1

    # 边界处理
    if allow_delta_zero:
        min_delta = 0
    else:
        min_delta = max(1, min_delta)
    max_delta = max(min_delta, min(max_delta, block_size - (0 if allow_delta_zero else 1)))

    if max_delta < min_delta:
        raise ValueError(f"无可用的 Δ 范围：min_delta={min_delta}, max_delta={max_delta}, block_size={block_size}")

    # 1) 生成/规范化 Δ
    if deltas is None:
        if method == "uniform":
            deltas = torch.randint(
                low=min_delta,
                high=max_delta + 1,
                size=(num_samples,),
                device=device,
                generator=generator,
            )
        elif method == "geometric":
            support = torch.arange(min_delta, max_delta + 1, device=device)
            shifted = support - (1 if not allow_delta_zero else 0)
            pmf = (geom_p * torch.pow(1 - geom_p, shifted - 1)).clamp_min(1e-12)
            pmf = pmf / pmf.sum()
            deltas = support[torch.multinomial(pmf, num_samples, replacement=True, generator=generator)]
        elif method == "mix":
            num_u = int(num_samples * uniform_frac)
            num_g = num_samples - num_u
            deltas_u = torch.randint(
                low=min_delta, high=max_delta + 1, size=(num_u,), device=device, generator=generator
            )
            support = torch.arange(min_delta, max_delta + 1, device=device)
            shifted = support - (1 if not allow_delta_zero else 0)
            pmf = (geom_p * torch.pow(1 - geom_p, shifted - 1)).clamp_min(1e-12)
            pmf = pmf / pmf.sum()
            deltas_g = support[torch.multinomial(pmf, num_g, replacement=True, generator=generator)]
            deltas = torch.empty(num_samples, device=device, dtype=torch.long)
            deltas[:num_u] = deltas_u
            deltas[num_u:] = deltas_g
            # 打乱
            perm = torch.randperm(num_samples, device=device, generator=generator)
            deltas = deltas[perm]
        else:
            raise ValueError(f"未知 method: {method}")
    else:
        deltas = deltas.to(device=device, dtype=torch.long)
        if (deltas < min_delta).any() or (deltas > max_delta).any():
            raise ValueError(f"deltas 超出范围 [{min_delta}, {max_delta}]")

        if deltas.numel() != num_samples:
            reps = (num_samples + deltas.numel() - 1) // deltas.numel()
            deltas = deltas.repeat(reps)[:num_samples]

    # 2) 对每个 Δ 采样左索引 i，使得 i ∈ [0, block_size - 1 - Δ]，然后 j = i + Δ
    i_max = (block_size - 1) - deltas    # [num_samples]
    span = i_max + 1                     # 每个样本的可选长度 > 0

    # 先采 [0,1) 浮点
    rand_u = torch.rand(num_samples, device=device, generator=generator)

    # 对 i 做左侧偏置：i_bias_power > 1 时，u^power 更靠近 0
    if i_bias_power != 1.0:
        # 为防止非法值，约束一下最小值
        i_bias_power_clamped = max(i_bias_power, 1.0)
        rand_u = rand_u.pow(i_bias_power_clamped)

    # 得到 [0, span_s) 的索引
    i_idx = (rand_u * span.to(rand_u.dtype)).floor().to(torch.long)
    j_idx = i_idx + deltas

    # （可选）安全检查
    # assert (i_idx >= 0).all()
    # assert (j_idx < block_size).all()

    return i_idx, j_idx, deltas
def compute_path_score_single(
    q_j: torch.Tensor,      # [H, D]
    k_i: torch.Tensor,      # [H, D]
    w: torch.Tensor,        # [T, H, D]
    beta: torch.Tensor,     # [T, H]
    i_idx: int,
    j_idx: int,
    *,
    sqrt_d_scale: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    assert q_j.dim() == 2 and k_i.dim() == 2, "q_j/k_i 应是 [H,D]"
    assert w.dim() == 3 and beta.dim() == 2, "w:[T,H,D], beta:[T,H]"
    H, D = q_j.shape
    T = w.shape[0]
    assert w.shape[1:] == (H, D) and beta.shape == (T, H)
    if not (0 <= i_idx < j_idx <= T - 1):
        raise ValueError(f"i/j 越界或 i>=j: i={i_idx}, j={j_idx}, T={T}")

    x = k_i.clone()                     # [H,D] 作为被变换的向量
    it = range(i_idx, j_idx)

    for t in it:
        w_t = w[t]                      # [H,D]
        b_t = beta[t].unsqueeze(-1)     # [H,1]
        dot = (x * w_t).sum(dim=-1, keepdim=True)  # [H,1] 逐 head 的 <x_h, w_th>
        x = x - b_t * dot * w_t         # x_h ← x_h - beta * <x_h,w_th> * w_th

    score_h = (q_j * x)     # 每个 head 的标量 [H,D]
    return score_h
def compute_path_score_multi(
    q: torch.Tensor,      # [B, T, H, D]
    k: torch.Tensor,      # [B, T, H, D]
    w: torch.Tensor,      # [B, T, H, D]
    beta: torch.Tensor,   # [B, T, H]
    i_idx: torch.Tensor,      # [S]
    j_idx: torch.Tensor,      # [S]
    batch_idx: torch.Tensor,  # [S]
) -> torch.Tensor:
    """
    返回: [S, H, D]，与 compute_path_score_single 的返回形状对齐（只是多了 S 维）
    """
    assert q.dim() == 4 and k.dim() == 4 and w.dim() == 4
    assert beta.dim() == 3
    B, T, H, D = q.shape
    assert k.shape == (B, T, H, D)
    assert w.shape == (B, T, H, D)
    assert beta.shape == (B, T, H)
    assert i_idx.shape == j_idx.shape == batch_idx.shape

    S = i_idx.shape[0]
    device = q.device

    # 合法性检查
    if not torch.all((0 <= i_idx) & (i_idx < j_idx) & (j_idx <= T - 1)):
        raise ValueError("i/j 越界或 i>=j")

    # 初始 x_s = k[b_s, i_s] 作为被变换的向量，形状 [S, H, D]
    x = k[batch_idx, i_idx, ...].clone()  # [S, H, D]

    # 对应的 q_j，形状 [S, H, D]
    q_j = q[batch_idx, j_idx, ...]        # [S, H, D]

    # 主循环只在时间轴 t 上扫一遍
    t_iter = range(T)


    for t in t_iter:
        # 这一时刻 t，哪些 sample 的区间 [i_s, j_s) 覆盖 t？
        # 条件: i_s <= t < j_s
        active = (i_idx <= t) & (t < j_idx)  # [S]
        if not torch.any(active):
            continue

        # 只更新 active 的那些样本
        b_t = batch_idx[active]              # [S_active]
        x_act = x[active]                    # [S_active, H, D]

        # 取对应 batch 上的 w_t, beta_t
        w_t = w[b_t, t, ...]                 # [S_active, H, D]
        beta_t = beta[b_t, t, ...].unsqueeze(-1)  # [S_active, H, 1]

        # dot = <x, w_t>，逐 head 内积
        dot = (x_act * w_t).sum(dim=-1, keepdim=True)  # [S_active, H, 1]

        # x ← x - beta * dot * w
        x[active] = x_act - beta_t * dot * w_t

    # 最后 score 与你原先一致：不在这里求和，只是逐维乘
    score = q_j * x  # [S, H, D]

    return score
def compute_pair_wavelet_scores_batched(
    q: torch.Tensor,                  # [sample_num, H, D]
    k: torch.Tensor,
    wavelet_decay_list: torch.Tensor, # [D,]  相对距离表
) -> torch.Tensor:

    _, _, D = q.shape
    assert wavelet_decay_list.dim()==2 and wavelet_decay_list.size(0)==D, "wavelet_decay_list 应为 [D,]"
    qk_scores = q * k   # [B,S,H,D]
    wavelet_scores = qk_scores * wavelet_decay_list.transpose(0, 1)[:, None, :] # [B,S,H,D] * [B,S,H,D] -> [B,S,H,D]
    return wavelet_scores

# ---------------------------
# helpers
# ---------------------------
def _match_heads(x: torch.Tensor, target_h: int) -> torch.Tensor:
    """
    x: [B,T,H,...]
    target_h: desired H
    repeat_interleave on head dim if needed (for GQA alignment)
    """
    H = x.shape[2]
    if H == target_h:
        return x
    if target_h % H == 0:
        return x.repeat_interleave(target_h // H, dim=2)
    raise ValueError(f"Cannot match heads: {H} -> {target_h}")

def _future_mask(T: int, device) -> torch.Tensor:
    # [1,1,T,T]
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1).view(1, 1, T, T)

# ---------------------------
# core: build A and compute M = lower(QW^T) T^{-1}
# ---------------------------
def path_ut_build_A(
    w: torch.Tensor,        # [B,T,H,d]
    beta: torch.Tensor,     # [B,T,H]
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    A = I + strictLower(W D W^T), unit-lower-triangular
    returns A: [B,H,T,T]
    """
    B, T, H, d = w.shape
    w0 = w.to(compute_dtype)
    b0 = beta.to(compute_dtype)

    eyeT = torch.eye(T, device=w.device, dtype=compute_dtype)

    w_scaled = w0 * b0.unsqueeze(-1)  # [B,T,H,d]
    S_wdw = torch.einsum("b i h d, b j h d -> b h i j", w_scaled, w0)  # [B,H,T,T]
    A = torch.tril(S_wdw, diagonal=-1) + eyeT.view(1, 1, T, T)         # unit-lower-tri
    return A
def _get_dist_ids():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
    return rank, local_rank, world_size, dev

@torch.no_grad()
def _quantiles_flat(x: torch.Tensor, qs=(0.1, 0.5, 0.9, 0.95, 0.99)):
    # flatten on CPU to reduce GPU overhead
    y = x.detach()
    if y.is_cuda:
        y = y.float().cpu()
    else:
        y = y.float()
    y = y.flatten()
    if y.numel() == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = torch.quantile(y, q).item()
    return out

@torch.no_grad()
def _router_gate_stats(
    *,
    router_name: str,
    logits: torch.Tensor,   # [B,T,H,S]
    gate: torch.Tensor,     # [B,T,H,S]
    tau: float,
    coe_for_rel: float,
    global_step: int,
    log_every: int,
    layer_idx: int,
    logger_obj=None,        # e.g. self.logger, or None->print
):
    if log_every <= 0:
        return
    if global_step < 0 or (global_step % log_every) != 0:
        return

    rank, local_rank, world_size, dev = _get_dist_ids()

    # -------- logits stats --------
    z = logits
    z_mean = z.float().mean().item()
    z_std  = z.float().std(unbiased=False).item()
    z_abs_p99 = _quantiles_flat(z.abs(), qs=(0.99,))["p99"]

    # norm over last dim S -> [B,T,H]
    z_norm = torch.linalg.vector_norm(z.float(), ord=2, dim=-1)
    z_norm_q = _quantiles_flat(z_norm, qs=(0.5, 0.9, 0.99))

    # -------- gate stats --------
    g = gate.float()
    eps = 1e-9
    ent = -(g * (g + eps).log()).sum(dim=-1)  # [B,T,H]
    top2, _ = torch.topk(g, k=2, dim=-1)
    top1 = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]

    S = g.shape[-1]
    idx = torch.arange(S, device=g.device, dtype=g.dtype)
    eshift = (g * idx).sum(dim=-1)  # [B,T,H]

    ent_mean = ent.mean().item()
    top1_mean = top1.mean().item()
    margin_mean = margin.mean().item()
    es_mean = eshift.mean().item()
    es_std = eshift.std(unbiased=False).item()

    ent_q = _quantiles_flat(ent, qs=(0.1, 0.5, 0.9))
    top1_q = _quantiles_flat(top1, qs=(0.5, 0.9, 0.99))
    margin_q = _quantiles_flat(margin, qs=(0.5, 0.9, 0.99))
    rel_co_val = coe_for_rel.item() if isinstance(coe_for_rel, torch.Tensor) else coe_for_rel
    msg = (
        f"[router stats] rank={rank}/{world_size} local_rank={local_rank} cuda={dev} "
        f"layer={layer_idx} step={global_step} router={router_name} tau={tau:.4g} rel_co={rel_co_val:.4g} | "
        f"logits mean={z_mean:.4e} std={z_std:.4e} abs_p99={z_abs_p99:.4e} "
        f"norm_p50={z_norm_q['p50']:.4e} norm_p90={z_norm_q['p90']:.4e} norm_p99={z_norm_q['p99']:.4e} | "
        f"entropy mean={ent_mean:.4f} p10={ent_q['p10']:.4f} p50={ent_q['p50']:.4f} p90={ent_q['p90']:.4f} | "
        f"top1 mean={top1_mean:.4f} p50={top1_q['p50']:.4f} p90={top1_q['p90']:.4f} p99={top1_q['p99']:.4f} | "
        f"margin mean={margin_mean:.4f} p50={margin_q['p50']:.4f} p90={margin_q['p90']:.4f} p99={margin_q['p99']:.4f} | "
        f"Eshift mean={es_mean:.4f} std={es_std:.4f}"
    )


    print(msg)

def path_ut_M_from_S(
    A: torch.Tensor,        # [B,H,T,T] unit-lower-tri
    S: torch.Tensor,        # [B,H,T,T] lower(..) already applied
    beta: torch.Tensor,     # [B,T,H]
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    M = S @ T^{-1}, where T^{-1} = A^{-1} D and D=diag(beta)
    We compute Z = S @ A^{-1} via A^{-T} trick, then M = Z @ D (column-wise scaling).
    returns M: [B,H,T,T]
    """
    b0 = beta.to(compute_dtype)
    beta_h = b0.transpose(1, 2)  # [B,H,T] (column index j)

    Zt = torch.linalg.solve_triangular(
        A.transpose(-1, -2),      # A^T (upper)
        S.transpose(-1, -2),      # RHS = S^T
        upper=True,
        unitriangular=True,
    )
    Z = Zt.transpose(-1, -2)      # [B,H,T,T] = S @ A^{-1}

    # right-multiply D => scale by beta on column j
    M = Z * beta_h.unsqueeze(2)   # [B,H,T,T]
    return M

def path_ut_base_raw(
    q: torch.Tensor,        # [B,T,H,d]
    k: torch.Tensor,        # [B,T,H,d]
    w: torch.Tensor,        # [B,T,H,d]
    beta: torch.Tensor,     # [B,T,H]
    compute_dtype: torch.dtype = torch.float32,
):
    """
    returns:
      E_base_raw: [B,H,T,T]   lower(QK^T) - M_base@strictLower(WK^T), NO mask, NO scale
      M_base:     [B,H,T,T]
      strict_WK:  [B,H,T,T]
      A:          [B,H,T,T]
    """
    B, T, H, d = w.shape
    q0 = q.to(compute_dtype)
    k0 = k.to(compute_dtype)
    w0 = w.to(compute_dtype)
    b0 = beta.to(compute_dtype)

    A = path_ut_build_A(w0, b0, compute_dtype=compute_dtype)

    QK = torch.einsum("b i h d, b j h d -> b h i j", q0, k0)
    WK = torch.einsum("b i h d, b j h d -> b h i j", w0, k0)

    lower_QK  = torch.tril(QK, diagonal=0)
    strict_WK = torch.tril(WK, diagonal=-1)

    QW = torch.einsum("b i h d, b j h d -> b h i j", q0, w0)
    S_base = torch.tril(QW, diagonal=0)

    M_base = path_ut_M_from_S(A, S_base, b0, compute_dtype=compute_dtype)

    E_base_raw = lower_QK - (M_base @ strict_WK)
    return E_base_raw, M_base, strict_WK, A

# ---------------------------
# optional: wavelet fused S_wave => M_wave (no OOM, d-chunk)
# ---------------------------
def path_ut_M_wave_fused(
    q: torch.Tensor,          # [B,T,H,d]
    w: torch.Tensor,          # [B,T,H,d]
    beta: torch.Tensor,       # [B,T,H]
    A: torch.Tensor,          # [B,H,T,T]
    wavelet_dtt: torch.Tensor,# [d,T,T]
    d_chunk: int = 8,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    S_wave[i,j] = lower( sum_d q[i,d] * w[j,d] * wavelet[d,i,j] )
    M_wave = S_wave @ T^{-1}
    returns M_wave: [B,H,T,T]
    """
    B, T, H, d = w.shape
    q0 = q.to(compute_dtype)
    w0 = w.to(compute_dtype)
    wav = wavelet_dtt.to(compute_dtype)
    b0 = beta.to(compute_dtype)

    S_wave = torch.zeros((B, H, T, T), device=q.device, dtype=compute_dtype)
    for d0 in range(0, d, d_chunk):
        d1 = min(d, d0 + d_chunk)
        qc = q0[..., d0:d1]      # [B,T,H,dc]
        wc = w0[..., d0:d1]      # [B,T,H,dc]
        wavc = wav[d0:d1]        # [dc,T,T]
        S_wave += torch.einsum("b i h c, b j h c, c i j -> b h i j", qc, wc, wavc)

    S_wave = torch.tril(S_wave, diagonal=0)
    M_wave = path_ut_M_from_S(A, S_wave, b0, compute_dtype=compute_dtype)
    return M_wave

# ---------------------------
# wavelet PE term using QH: rel = (QH) P^T, where QH = Q - (M W)

def wavelet_rel_from_M_bands(
    q: torch.Tensor,              # [B,T,H,d]
    w: torch.Tensor,              # [B,T,H,d]
    M: torch.Tensor,              # [B,H,T,T]
    wavelet_dtt_bands: torch.Tensor,  # [K,d,T,T]   (K=BANDS=8)
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    returns rel_bands: [B,H,K,T,T]  (NO scale, NO mask)
    """
    q0 = q.to(compute_dtype)
    w0 = w.to(compute_dtype)
    wav = wavelet_dtt_bands.to(compute_dtype)  # [K,d,T,T]

    # rel_1[k] = Q P_k^T
    rel_1 = torch.einsum("b t h d, k d t n -> b h k t n", q0, wav)

    # q_corr = (M W)
    q_corr = torch.einsum("b h t j, b j h d -> b t h d", M, w0)

    # rel_2[k] = (M W) P_k^T
    rel_2 = torch.einsum("b t h d, k d t n -> b h k t n", q_corr, wav)

    rel_bands = rel_1 - rel_2
    return rel_bands
def wavelet_rel_from_M(
    q: torch.Tensor,            # [B,T,H,d]
    w: torch.Tensor,            # [B,T,H,d]
    M: torch.Tensor,            # [B,H,T,T]
    wavelet_dtt: torch.Tensor,  # [d,T,T]
    compute_dtype: torch.dtype = torch.float32,
    layer_idx: int = None,
    rel_selection = None,
    E_base_raw: torch.Tensor = None,
) -> torch.Tensor:
    """
    rel = (QH) P^T = Q P^T - (M W) P^T
    returns rel: [B,H,T,T]  (NO scale, NO mask)
    """
    q0 = q.to(compute_dtype)
    w0 = w.to(compute_dtype)
    wav = wavelet_dtt.to(compute_dtype)
    rel1_coe, rel2_coe = 1.0, 1.0
    if rel_selection == 'rel1':
        rel_1 = torch.einsum("b t h d, d t n -> b h t n", q0, wav)  # Q P^T
        rel = rel1_coe * rel_1
    elif rel_selection == 'rel2':
        q_corr = torch.einsum("b h t j, b j h d -> b t h d", M, w0) # (M W)
        rel_2  = torch.einsum("b t h d, d t n -> b h t n", q_corr, wav)
        rel = -rel2_coe * rel_2
    elif rel_selection == 'all':
        q_corr = torch.einsum("b h t j, b j h d -> b t h d", M, w0) # (M W)
        # rel = rel2_coe * (torch.einsum("b t h d, d t n -> b h t n", q0, wav) - rel2_coe * torch.einsum("b t h d, d t n -> b h t n", q_corr, wav))
        rel = rel1_coe * torch.einsum("b t h d, d t n -> b h t n", q0, wav) - rel2_coe * torch.einsum("b t h d, d t n -> b h t n", q_corr, wav)


    return rel
def causal_mask_fill_value(dtype: torch.dtype) -> float:
    # 对 fp16/bf16/float32 都安全
    return torch.finfo(dtype).min
# ---------------------------
# final: baseline output + wavelet(QH) output
# ---------------------------
import torch

def wavelet_rel_from_M_stream(
    q: torch.Tensor,              # [B,T,H,d]
    w: torch.Tensor,              # [B,T,H,d]
    M: torch.Tensor,              # [B,H,T,T]
    wavelet_dtt_bands: torch.Tensor,  # [K,d,T,T]
    alpha: torch.Tensor | None = None, # [B,H,K] or None (uniform)
    compute_dtype: torch.dtype = torch.bfloat16,
    n_chunk: int = 128,           # chunk over last dim of [T,T]
):
    """
    Returns rel: [B,H,T,T] (NO scale, NO mask), streaming over K and n_chunk.
    Peak memory ~ O(B*H*T*n_chunk) instead of O(B*H*K*T*T).
    """
    B, T, H, d = q.shape
    K, d2, T1, T2 = wavelet_dtt_bands.shape
    assert d2 == d and T1 == T and T2 == T, (q.shape, wavelet_dtt_bands.shape)

    q0 = q.to(compute_dtype)
    w0 = w.to(compute_dtype)
    wav = wavelet_dtt_bands.to(compute_dtype)

    # q_corr = (M W)  -> [B,T,H,d]
    q_corr = torch.einsum("b h t j, b j h d -> b t h d", M, w0)

    rel = torch.zeros((B, H, T, T), device=q.device, dtype=compute_dtype)

    if alpha is not None:
        assert alpha.shape == (B, H, K), (alpha.shape, (B, H, K))
        a = alpha.to(device=q.device, dtype=compute_dtype)
    else:
        a = None

    for k in range(K):
        wav_k = wav[k]  # [d,T,T]
        wgt = a[:, :, k] if a is not None else None  # [B,H] or None

        for n0 in range(0, T, n_chunk):
            n1 = min(T, n0 + n_chunk)

            # rel1_chunk: Q P_k^T
            rel1 = torch.einsum("b t h d, d t n -> b h t n", q0, wav_k[:, :, n0:n1])
            # rel2_chunk: (MW) P_k^T
            rel2 = torch.einsum("b t h d, d t n -> b h t n", q_corr, wav_k[:, :, n0:n1])

            chunk = rel1 - rel2
            if wgt is not None:
                chunk = chunk * wgt[:, :, None, None]

            rel[:, :, :, n0:n1] += chunk

    return rel



class PathToWaveletRouter(torch.nn.Module):
    def __init__(self, head_dim_feat: int, num_heads: int, num_bands: int, hidden: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(head_dim_feat, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, num_bands),
        )

    def forward(self, feat_h: torch.Tensor, temperature: float = 1.0):
        """
        feat_h: [B, H, F]  (每个 head 的特征)
        return alpha: [B, H, B]  (band mixing)
        """
        logits = self.mlp(feat_h) / temperature
        alpha = torch.softmax(logits, dim=-1)
        return alpha
class LogitsToBandRouter(nn.Module):
    """
    Input : E_base_raw [B,H,T,T]
    Output: alpha      [B,H,K]
    """
    def __init__(self, num_heads: int, num_bands: int, hidden: int = 64, pool: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.pool = pool

        # depthwise conv per head (keeps heads independent)
        self.dw = nn.Conv2d(num_heads, num_heads, kernel_size=3, padding=1, groups=num_heads)
        # pointwise conv: each head -> hidden channels (still grouped by head)
        self.pw = nn.Conv2d(num_heads, num_heads * hidden, kernel_size=1, groups=num_heads)

        self.out = nn.Linear(hidden, num_bands)

    def forward(self, E_base_raw: torch.Tensor, temperature: float = 1.0):
        assert E_base_raw.dim() == 4, E_base_raw.shape
        B, H, T, _ = E_base_raw.shape
        assert H == self.num_heads, (H, self.num_heads)

        x = E_base_raw  # [B,H,T,T]

        # downsample for speed (T=512 -> 64 if pool=8)
        if self.pool > 1:
            x = F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)  # [B,H,Ts,Ts]

        x = self.dw(x)                       # [B,H,Ts,Ts]
        x = self.pw(x)                       # [B,H*hidden,Ts,Ts]

        Ts = x.shape[-1]
        x = x.view(B, H, -1, Ts, Ts).mean(dim=(-1, -2))  # [B,H,hidden]  (全局聚合)

        logits = self.out(x) / temperature   # [B,H,K]
        alpha = torch.softmax(logits, dim=-1)
        return alpha

class PaTHAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        use_forget_gate: bool = False,
        use_qk_norm: bool = False,
        layer_idx: int = None,
        use_low_rank_w: bool = True,
        use_w_shortconv: bool = True,
        conv_size: int = 3,
        conv_bias: bool = False,
        # NEW ↓↓↓
        num_harmonics: int = 1,   # rank 数，=1 退化为原版
        use_wavelet_beta: bool = False,
        wavelet_mode: str = "additive",   # "additive" | "softmix"
        logging_steps: int = 1000,
        wavelet_baseline_use: bool = False,
        attn_pdrop=0.1,
        init_theta=0.847,   # initial theta for path attention ratio
        use_soft_wavelet_fox=False,
        config=None,
    ):
        super().__init__()

        self._debug_enabled = False   # eval 开始由 callback 打开
        self._debug_probe_done = False
        self.debug_accum = {}         # layer_idx -> stats dict
        self._eval_bin_stats = {}
        self._eval_batch_step = 0
        self._eval_stats_logged_once = False

        self.eval_stats_enabled = self._as_bool(getattr(config, "eval_rel_stats_enabled", True), default=True)
        self.eval_stats_layers = self._parse_layer_set(getattr(config, "eval_rel_stats_layers", "0"))
        self.eval_stats_bin_size = max(1, int(getattr(config, "eval_rel_stats_bin_size", 256)))
        self.eval_stats_log_every = max(0, int(getattr(config, "eval_rel_stats_log_every", 0)))
        self.eval_stats_log_once = self._as_bool(getattr(config, "eval_rel_stats_log_once", True), default=True)
        self.eval_stats_per_head = self._as_bool(getattr(config, "eval_rel_stats_per_head", False), default=False)
        self.eval_stats_eps = float(getattr(config, "eval_rel_stats_eps", 1e-6))
        self.eval_stats_max_samples_per_bin = max(
            128, int(getattr(config, "eval_rel_stats_max_samples_per_bin", 4096))
        )
        self.eval_stats_anchor_layer = int(getattr(config, "eval_rel_stats_anchor_layer", 0))
        self.rel_alpha = float(getattr(config, "rel_alpha", getattr(config, "attn_rel_alpha", 1.0)))
        self.log_rel_stats = self._as_bool(getattr(config, "log_rel_stats", False), default=False)
        self.log_rel_every = max(0, int(getattr(config, "log_rel_every", 500)))
        self.log_rel_eval_every = max(0, int(getattr(config, "log_rel_eval_every", 0)))
        self.log_rel_tail_tau = max(1, int(getattr(config, "log_rel_tail_tau", 1024)))
        qpos_cfg = getattr(config, "log_rel_sample_qpos", None)
        if qpos_cfg is None:
            qpos_cfg = getattr(config, "log_rel_sample_tokens", "128,512,2048")
        self.log_rel_sample_qpos = self._parse_int_list(qpos_cfg, default=[128, 512, 2048])
        self.log_rel_sample_heads = self._parse_int_list(
            getattr(config, "log_rel_sample_heads", "0,3,7,11"),
            default=[0, 3, 7, 11],
        )
        self.log_rel_sample_key_offsets = self._parse_int_list(
            getattr(config, "log_rel_sample_key_offsets", "0,16,64,256,1024"),
            default=[0, 16, 64, 256, 1024],
        )
        self.log_rel_sample_batch_idx = max(0, int(getattr(config, "log_rel_sample_batch_idx", 0)))
        # Eval rel stats are opt-in; callback flips this flag for selected eval rounds.
        self._rel_eval_collect = False
        self._rel_debug_cache = {}
        self._rel_train_hook_buffer = {}
        self._rel_eval_buffer = {}
        self._rel_last_raw_logits = None
        self._rel_last_coe_value = None
        debug_attn_margin_cfg = getattr(config, "debug_attn_margin", None)
        if debug_attn_margin_cfg is None:
            debug_attn_margin_cfg = getattr(config, "attn_margin_stats_enabled", False)
        self.attn_margin_enabled = self._as_bool(debug_attn_margin_cfg, default=False)
        self.attn_margin_layers = self._parse_layer_set(getattr(config, "attn_margin_stats_layers", "all"))
        # Keep margin logger bins aligned with EvalStats bins.
        self.attn_margin_bin_size = int(self.eval_stats_bin_size)
        self.attn_margin_log_every = max(
            0, int(getattr(config, "attn_margin_stats_log_every", getattr(config, "debug_attn_margin_log_every", 500)))
        )
        self.attn_margin_log_per_head = self._as_bool(
            getattr(config, "attn_margin_stats_log_per_head", False), default=False
        )
        self.attn_margin_log_head_limit = max(
            0, int(getattr(config, "attn_margin_stats_log_head_limit", 0))
        )
        self.attn_margin_log_perplexity = self._as_bool(
            getattr(config, "attn_margin_stats_log_perplexity", False), default=False
        )
        self.attn_margin_train_enabled = self._as_bool(
            getattr(config, "attn_margin_stats_train_enabled", True), default=True
        )
        self.attn_margin_eval_enabled = self._as_bool(
            getattr(config, "attn_margin_stats_eval_enabled", True), default=True
        )
        self._attn_margin_local_step = 0

        self.router_norm_cfg = build_router_norm_config(config)
        self._router_norm_local_step = 0
        self.router_norm = None
        self.router_norm_logger = None
        if self.router_norm_cfg.enable:
            norm_dim = getattr(config, "router_band_num", None) if config is not None else None
            try:
                if bool(getattr(config, "hierarchical_gate_use", False)):
                    d_chunk_cfg = int(getattr(config, "router_d_chunk", 8))
                    head_dim_local = int(hidden_size) // int(num_heads)
                    if d_chunk_cfg > 0 and (head_dim_local % d_chunk_cfg == 0):
                        norm_dim = head_dim_local // d_chunk_cfg
            except Exception:
                pass
            if norm_dim is not None:
                try:
                    norm_dim = int(norm_dim)
                    if norm_dim <= 0:
                        norm_dim = None
                except Exception:
                    norm_dim = None
            self.router_norm = RouterNorm(
                norm_type=self.router_norm_cfg.norm_type,
                eps=self.router_norm_cfg.eps,
                clamp_std_min=self.router_norm_cfg.clamp_std_min,
                affine=self.router_norm_cfg.affine,
                feature_dim=norm_dim,
            )
            self.router_norm_logger = RouterNormStatsLogger(
                log_every=self.router_norm_cfg.log_every,
                log_heads=self.router_norm_cfg.log_heads,
                log_tokens=self.router_norm_cfg.log_tokens,
            )

        self.tau = getattr(config, "tau", 1.0)
        self._coe_for_rel_init = float(getattr(config, "coe_for_rel_init", -1))
        self.rel_coe = float(getattr(config, "coe_for_rel", 1.0))
        if self._coe_for_rel_init != -1:
            self.coe_for_rel = nn.Parameter(torch.tensor(self.rel_coe, dtype=torch.float32))
            self.reset_parameters()
        else:
            self.coe_for_rel = self.rel_coe
        # logging / steps
        self.config= config
        self.logging_steps = 1000
        self.steps = 0
        self.total_steps = 100000
        self.use_wavelet_beta = use_wavelet_beta
        self.wavelet_mode = wavelet_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.r = int(num_harmonics)
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.layer_idx = layer_idx
        # Router 局部步数计数（未传全局步数时作为 fallback，每层/每卡独立）
        self.router_local_step = 0
        # 如果外部没传递累积步数，默认不做折算
        self.router_grad_accum_steps = getattr(config, "gradient_accumulation_steps", 1)
        if config.distill_teacher == 'mean_wavelet_pe':
            if config.wavelet_pe_softmax_use:
                load_spec_teacher = torch.load(f"/cl/work5/hongyu-s/gpt2_test/transformers/examples/pytorch/language-modeling/spectra_wavelet_teacher/layer_{layer_idx}_spectrum.pt")
            else:
                load_spec_teacher = torch.load(f"/cl/work5/hongyu-s/gpt2_test/transformers/examples/pytorch/language-modeling/wo_softmax_spectra_wavelet_teacher/layer_{layer_idx}_spectrum.pt")
            teacher_mean_spec = load_spec_teacher["mean_spectrum"]
            self.register_buffer("teacher_mean_spec", teacher_mean_spec)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.rel1_coe, self.wavelet_coe = None, None
        # try:
        #     if str(getattr(config, "coe_mode", "")).lower() == "seperate":
        #         self.rel1_coe = nn.Parameter(torch.ones(1, 12, 1, 1), requires_grad=True)
        #         self.wavelet_coe = nn.Parameter(torch.ones(1, 12, 1, 1), requires_grad=True)
        #     elif str(getattr(config, "coe_mode", "")).lower() == "unify":
        #         self.rel1_coe = None
        #         self.wavelet_coe = nn.Parameter(torch.zeros(1, 12, 1, 1), requires_grad=True)
        #     elif str(getattr(config, "coe_mode", "")).lower() == "none":
        #         self.rel1_coe = None
        #         self.wavelet_coe = None
        # except:
        #     self.rel1_coe = nn.Parameter(torch.zeros(1, 12, 1, 1), requires_grad=True)
        #     self.wavelet_coe = nn.Parameter(torch.ones(1, 12, 1, 1), requires_grad=True)
        # w 分支输出扩到 H*R*d
        out_w = self.kv_dim * self.r
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w, bias=False)
            )
            rel_use = self._rel_layer_enabled(layer_idx, config)
            if config.wavelet_router and rel_use:
                if getattr(config, "hierarchical_gate_use", False):
                    d_chunk = getattr(config, "router_d_chunk", 8)
                    assert self.head_dim % d_chunk == 0
                    S = self.head_dim // d_chunk

                    # local: token-wise, per-head, over S
                    self.local_router1  = nn.Linear(self.head_dim, S, bias=False)
                    self.local_router2  = nn.Linear(self.head_dim, S, bias=False)

                    # global: sequence-wise pooled, per-head, over S
                    self.global_router1 = nn.Linear(self.hidden_size, S, bias=False)
                    self.global_router2 = nn.Linear(self.hidden_size, S, bias=False)
                else:
                    if config.router_mode == 'unify':
                        self.router1 = _make_router_mlp(
                            self.hidden_size,
                            self.num_heads * self.config.router_band_num,
                            bool(getattr(config, "router_non_linear_use", False)),
                        )
                        self.router2 = None
                    elif config.router_mode == 'seperate':
                        if config.router_gate_use:
                            self.low_rank_map1 = nn.Linear(self.hidden_size, 32, bias=False)
                            self.low_rank_map2 = nn.Linear(self.hidden_size, 32, bias=False)

                            # scale logits heads
                            self.router1_head = nn.Linear(32, self.num_heads * self.config.router_band_num, bias=False)
                            self.router2_head = nn.Linear(32, self.num_heads * self.config.router_band_num, bias=False)

                            # use gates (IMPORTANT: bias=True)
                            self.router1_gate_head = nn.Linear(32, self.num_heads, bias=True)
                            self.router2_gate_head = nn.Linear(32, self.num_heads, bias=True)
                            self.tau_router = 1.0
                            self.t_gate = 1.0

                            with torch.no_grad():
                                self.router1_gate_head.bias.fill_(2.197)
                                self.router2_gate_head.bias.fill_(2.197)
                        else:                 
                            router_map_layer_num = getattr(config, "router_map_layer_num", 2)
                            if router_map_layer_num == 2:
                                self.router1 = _make_router_mlp(
                                    self.hidden_size,
                                    self.num_heads * self.config.router_band_num,
                                    bool(getattr(config, "router_non_linear_use", False)),
                                )
                                self.router2 = _make_router_mlp(
                                    self.hidden_size,
                                    self.num_heads * self.config.router_band_num,
                                    bool(getattr(config, "router_non_linear_use", False)),
                                )
                                # print('router_non_linear_use', getattr(config, "router_non_linear_use", False))
                            elif router_map_layer_num == 1:
                                self.router1 = nn.Linear(self.hidden_size, self.num_heads * self.config.router_band_num, bias=False)
                                self.router2 = nn.Linear(self.hidden_size, self.num_heads * self.config.router_band_num, bias=False)
                            else:
                                raise ValueError(f"Unknown router_map_layer_num: {router_map_layer_num}")
            else:
                self.router1 = None
                self.router2 = None
        else:
            self.w_proj = nn.Linear(self.hidden_size, out_w, bias=False)

        # Q/K 归一
        if use_qk_norm:
            self.maybe_q_norm = RMSNorm(self.hidden_size)
            self.maybe_k_norm = RMSNorm(self.kv_dim)
        else:
            self.maybe_q_norm = nn.Identity()
            self.maybe_k_norm = nn.Identity()

        # depthwise short conv 对拼接后的 H*R*d
        self.use_w_shortconv = use_w_shortconv
        if use_w_shortconv:
            self.w_conv1d = ShortConvolution(
                hidden_size=out_w, kernel_size=conv_size, bias=conv_bias, activation='silu'
            )

        # 每个 (head, rank) 一个 beta
        self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.r, bias=True)
        if self.config.distill_teacher == 'rotary' or self.config.qk_rotation:
            self.rotary_emb = RotaryEmbedding(dim=64)

        # 可选 FoX 遗忘门
        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        # 输出层（H*R*d → hidden_size）
        self.o_proj = nn.Linear(self.hidden_size * self.r, self.hidden_size, bias=False)
        self.wavelet_baseline_use = wavelet_baseline_use
        if wavelet_baseline_use:
            self.attn_dropout = nn.Dropout(attn_pdrop)
            self.path_attention_ratio = nn.Parameter(torch.ones(num_heads)) 
        # ===== Wavelet(beta) 参数 =====
        # if config.wavelet_router:
        #     self.router_module = LogitsToBandRouter(
        #         num_heads=self.num_heads,
        #         num_bands=config.router_band_num,
        #         hidden=config.router_hidden_dim,
        #         pool=8,
        #     )
        # else:
        #     self.router_module = None
        if use_wavelet_beta:
            H = self.num_kv_heads

            # 1) 你的新要求：指数项可学，且初始化为负数序列
            #    e = [-2*(h//2) for h in range(H)]  → [0,0,-2,-2,-4,-4,...,-10,-10] 当 H=12
            # exp_list   = [-2 * (h // 2) for h in range(H)]
            # exp_list = [-1e6] * H
            exp_list = [0] * H
            shift_list = [float(h % 2) for h in range(H)]  # [0,1,0,1,...]

            self.ricker_scale_exp   = torch.tensor(exp_list, dtype=torch.float32, device='cuda').unsqueeze(1)  # [H,1]
            self.ricker_shift = torch.tensor(shift_list, dtype=torch.float32, device='cuda').unsqueeze(1)  # [H,1]

            # exp_init   = torch.tensor(exp_list, dtype=torch.float32).view(1,1,H,1).repeat(1,1,1,self.r)
            # shift_init = torch.tensor(shift_list, dtype=torch.float32).view(1,1,H,1).repeat(1,1,1,self.r)

            # ★ e 是可学习参数；scale = 2**e 在 forward 里计算
            # self.ricker_scale_exp = nn.Parameter(exp_init)        # [1,1,H,r]
            # self.ricker_shift     = nn.Parameter(shift_init)      # [1,1,H,r]

            # 振幅：从 0 起，不扰动基线
            # self.ricker_amp = nn.Parameter(torch.ones(1, 1, H, self.r, dtype=torch.float32))

            # softmix 的可学习混合系数
            # self.mix_logit = nn.Parameter(torch.tensor(1.0)) if wavelet_mode == "softmix" else None

    # 小工具：按 head 打印
    def _get_layer_accum(self, layer_idx: int):
        st = self.debug_accum.get(layer_idx)
        if st is None:
            st = {
                "sum_abs_rel": 0.0,
                "sum_abs_eb":  0.0,
                "sum_batch_ratio": 0.0,
                "sum_elem_ratio": 0.0,
                "sum_entropy": 0.0,   # attention entropy (not router entropy)
                "sum_top1":    0.0,   # attention top1
                "count":       0.0,
                # tail quantiles: keep small reservoir sample (float32)
                "samples_rel_abs": [],
                "samples_eb_abs": [],
                "samples_batch_ratio": [],
                "samples_elem_ratio": [],
                "samples_attn_top1": [],
                "samples_router_top1": [],
                "samples_router_margin": [],
                # sequence-length buckets for extended-length comparison
                "by_seq_len": {},
                "eval_bin_state": None,
            }
            self.debug_accum[layer_idx] = st
        return st

    @staticmethod
    def _as_bool(v, default=False):
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            x = v.strip().lower()
            if x in ("1", "true", "yes", "y", "on"):
                return True
            if x in ("0", "false", "no", "n", "off"):
                return False
        return bool(default)

    @staticmethod
    def _parse_layer_set(v):
        if v is None:
            return {0}
        if isinstance(v, str):
            x = v.strip().lower()
            if x in ("all", "*"):
                return None
            items = [p.strip() for p in v.split(",") if p.strip()]
            out = set()
            for it in items:
                try:
                    out.add(int(it))
                except Exception:
                    continue
            return out if out else {0}
        if isinstance(v, (list, tuple, set)):
            out = set()
            for it in v:
                try:
                    out.add(int(it))
                except Exception:
                    continue
            return out if out else {0}
        try:
            return {int(v)}
        except Exception:
            return {0}

    def _track_eval_layer(self, layer_idx: int) -> bool:
        if not self.eval_stats_enabled:
            return False
        if self.eval_stats_layers is None:
            return True
        return int(layer_idx) in self.eval_stats_layers

    def _is_eval_anchor_layer(self, layer_idx: int) -> bool:
        lid = int(layer_idx)
        if self.eval_stats_layers and self.eval_stats_anchor_layer not in self.eval_stats_layers:
            return lid == min(self.eval_stats_layers)
        return lid == self.eval_stats_anchor_layer

    @staticmethod
    def _to_int_or_none(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                return None
            x = x.detach().item()
        try:
            return int(x)
        except Exception:
            return None

    @staticmethod
    def _parse_int_list(v, default=None):
        if default is None:
            default = []
        out = []
        if v is None:
            src = list(default)
        elif isinstance(v, str):
            s = v.strip()
            if not s:
                src = list(default)
            else:
                s = s.strip("[]()")
                src = []
                for tok in re.split(r"[,\s]+", s):
                    if tok:
                        src.append(tok)
        elif isinstance(v, (list, tuple, set)):
            src = list(v)
        else:
            src = [v]
        for item in src:
            try:
                out.append(int(item))
            except Exception:
                continue
        if out:
            return out
        return [int(x) for x in default]

    def _get_rel_layer_id(self, layer_idx: Optional[int]) -> int:
        if layer_idx is None:
            return int(self.layer_idx if self.layer_idx is not None else -1)
        return int(layer_idx)

    @staticmethod
    def _parse_rel_layer_set(v):
        """Return None for all-layers, otherwise a set[int] of enabled layers."""
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "all", "*"):
                return None
            s = s.strip("[]()")
            tokens = [tok for tok in re.split(r"[,\s]+", s) if tok]
        elif isinstance(v, (list, tuple, set)):
            tokens = list(v)
        else:
            tokens = [v]
        out = set()
        for tok in tokens:
            try:
                out.add(int(tok))
            except Exception:
                continue
        return out if out else None

    def _rel_layer_enabled(self, layer_idx: Optional[int], config=None) -> bool:
        lid = self._get_rel_layer_id(layer_idx)
        cfg_obj = config if config is not None else getattr(self, "config", None)
        rel_layers_raw = getattr(cfg_obj, "rel_use_layer_list", None) if cfg_obj is not None else None
        enabled = self._parse_rel_layer_set(rel_layers_raw)
        if enabled is None:
            return True
        return int(lid) in enabled

    @staticmethod
    def _rel_to_tensor(v, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(v):
            return v.to(device=ref.device, dtype=ref.dtype)
        if v is None:
            return torch.tensor(1.0, device=ref.device, dtype=ref.dtype)
        return torch.tensor(float(v), device=ref.device, dtype=ref.dtype)

    def _rel_should_log_train(self, global_step) -> tuple[bool, Optional[int]]:
        if not self.log_rel_stats:
            return False, None
        if not self.training:
            return False, None
        if not torch.is_grad_enabled():
            return False, None
        if self.log_rel_every <= 0:
            return False, None
        step = self._to_int_or_none(global_step)
        if step is None:
            return False, None
        step_eff = int(step) + 1
        return (step_eff % int(self.log_rel_every) == 0), step_eff

    def _rel_should_log_eval(self) -> bool:
        return (
            bool(self.log_rel_stats)
            and int(getattr(self, "log_rel_eval_every", 0)) > 0
            and (not self.training)
            and bool(getattr(self, "_rel_eval_collect", False))
        )

    def _build_rel_sample_spec(self, T: int, H: int, device: torch.device):
        if T <= 0 or H <= 0:
            return None
        heads = []
        seen_heads = set()
        src_heads = self.log_rel_sample_heads if self.log_rel_sample_heads else [0]
        for h_raw in src_heads:
            h = int(h_raw)
            if h < 0:
                h = H + h
            h = max(0, min(H - 1, h))
            if h not in seen_heads:
                seen_heads.add(h)
                heads.append(h)
        if not heads:
            heads = [0]

        queries = []
        seen_queries = set()
        src_queries = self.log_rel_sample_qpos if self.log_rel_sample_qpos else [T - 1]
        for q_raw in src_queries:
            q = int(q_raw)
            if q < 0:
                q = T + q
            q = max(0, min(T - 1, q))
            if q not in seen_queries:
                seen_queries.add(q)
                queries.append(q)
        if not queries:
            queries = [T - 1]

        pair_offsets = self.log_rel_sample_key_offsets if self.log_rel_sample_key_offsets else [0, 16, 64, 256, 1024]
        pairs = []
        seen_pairs = set()
        for m in queries:
            for off in pair_offsets:
                n = m - int(off)
                if n < 0 or n > m:
                    continue
                key = (int(m), int(n))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(key)
        if not pairs:
            pairs = [(queries[-1], queries[-1])]

        m_idx = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        n_idx = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        h_idx = torch.tensor(heads, device=device, dtype=torch.long)
        q_idx = torch.tensor(queries, device=device, dtype=torch.long)
        return {
            "head_idx": h_idx,
            "q_idx": q_idx,
            "m_idx": m_idx,
            "n_idx": n_idx,
            "query_positions": queries,
        }

    @staticmethod
    def _sample_rel_pairs(x: torch.Tensor, sample_spec: dict) -> torch.Tensor:
        y = x.index_select(1, sample_spec["head_idx"])
        return y[:, :, sample_spec["m_idx"], sample_spec["n_idx"]]

    def _sample_rel_pairs_b0(self, x: torch.Tensor, sample_spec: dict) -> torch.Tensor:
        b = min(int(self.log_rel_sample_batch_idx), int(x.shape[0]) - 1)
        y = x[b : b + 1].index_select(1, sample_spec["head_idx"])
        return y[:, :, sample_spec["m_idx"], sample_spec["n_idx"]]

    def _sample_query_rows_b0(self, x: torch.Tensor, sample_spec: dict) -> torch.Tensor:
        b = min(int(self.log_rel_sample_batch_idx), int(x.shape[0]) - 1)
        y = x[b : b + 1].index_select(1, sample_spec["head_idx"]).index_select(2, sample_spec["q_idx"])
        return y

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        t = x.detach().float().reshape(-1)
        if t.numel() == 0:
            return float("nan")
        return float(torch.sqrt((t * t).mean()).item())

    @staticmethod
    def _corr_flat(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        xv = x.detach().float().reshape(-1)
        yv = y.detach().float().reshape(-1)
        if xv.numel() == 0 or yv.numel() == 0 or xv.numel() != yv.numel():
            return float("nan")
        xv = xv - xv.mean()
        yv = yv - yv.mean()
        den = torch.sqrt((xv * xv).sum()) * torch.sqrt((yv * yv).sum())
        if float(den.item()) <= eps:
            return float("nan")
        return float((xv * yv).sum().div(den.clamp_min(eps)).item())

    @staticmethod
    @torch.no_grad()
    def _rel_summary_stats(x: torch.Tensor) -> dict:
        t = x.detach().float().reshape(-1)
        if t.numel() == 0:
            nan = float("nan")
            return {
                "mean": nan,
                "abs_mean": nan,
                "std": nan,
                "min": nan,
                "max": nan,
                "p90": nan,
                "p99": nan,
            }
        q = torch.quantile(t, torch.tensor([0.9, 0.99], device=t.device, dtype=t.dtype))
        return {
            "mean": float(t.mean().item()),
            "abs_mean": float(t.abs().mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "p90": float(q[0].item()),
            "p99": float(q[1].item()),
        }

    @torch.no_grad()
    def _rel_attn_shape_stats(self, attn_probs: torch.Tensor, sample_spec: dict, tau: int) -> dict:
        p = self._sample_query_rows_b0(attn_probs.detach().float(), sample_spec)
        ent_vals, top1_vals, tail_vals = [], [], []
        eps = 1e-12
        for qi, m in enumerate(sample_spec["query_positions"]):
            m = int(m)
            if qi < 0 or qi >= p.shape[2]:
                continue
            valid = p[:, :, qi, : m + 1]
            valid = valid / valid.sum(dim=-1, keepdim=True).clamp_min(eps)
            ent = -(valid * valid.clamp_min(eps).log()).sum(dim=-1)
            top1 = valid.max(dim=-1).values
            if m >= tau:
                tail = valid[..., : (m - tau + 1)].sum(dim=-1)
            else:
                tail = torch.zeros_like(top1)
            ent_vals.append(float(ent.mean().item()))
            top1_vals.append(float(top1.mean().item()))
            tail_vals.append(float(tail.mean().item()))

        if not ent_vals:
            return {"ent": float("nan"), "top1": float("nan"), "tail": float("nan")}
        return {
            "ent": float(sum(ent_vals) / len(ent_vals)),
            "top1": float(sum(top1_vals) / len(top1_vals)),
            "tail": float(sum(tail_vals) / len(tail_vals)),
        }

    @torch.no_grad()
    def _rel_record_train_hook(
        self,
        step: int,
        grad_total_logits: torch.Tensor,
        rel_logits_tensor: torch.Tensor,
        sample_spec: dict,
        coe_value_tensor: torch.Tensor,
    ):
        grad_s = self._sample_rel_pairs_b0(grad_total_logits.detach(), sample_spec).float().reshape(-1)
        rel_s = self._sample_rel_pairs_b0(rel_logits_tensor.detach(), sample_spec).float().reshape(-1)
        if grad_s.numel() == 0 or rel_s.numel() == 0:
            return
        prod = grad_s * rel_s
        corr = self._corr_flat(grad_s, rel_s)
        bucket = self._rel_train_hook_buffer.setdefault(
            int(step),
            {
                "n_elem": 0.0,
                "sum_A": 0.0,
                "sum_abs_grad": 0.0,
                "sum_grad": 0.0,
                "sum_grad_sq": 0.0,
                "max_abs_grad": 0.0,
                "sum_coe": 0.0,
                "coe_count": 0.0,
                "sum_corr_grad_rel": 0.0,
                "corr_count": 0.0,
            },
        )
        bucket["n_elem"] += float(grad_s.numel())
        bucket["sum_A"] += float(prod.sum().item())
        bucket["sum_abs_grad"] += float(grad_s.abs().sum().item())
        bucket["sum_grad"] += float(grad_s.sum().item())
        bucket["sum_grad_sq"] += float((grad_s * grad_s).sum().item())
        bucket["max_abs_grad"] = max(float(bucket["max_abs_grad"]), float(grad_s.abs().max().item()))
        coe_mean = float(coe_value_tensor.detach().float().mean().item())
        bucket["sum_coe"] += coe_mean
        bucket["coe_count"] += 1.0
        if math.isfinite(corr):
            bucket["sum_corr_grad_rel"] += float(corr)
            bucket["corr_count"] += 1.0

    @torch.no_grad()
    def _rel_accumulate_eval(
        self,
        layer_id: int,
        sample_spec: dict,
        base_logits_detached: torch.Tensor,
        rel_logits_detached: torch.Tensor,
        rel_effective_detached: torch.Tensor,
        total_logits_detached: torch.Tensor,
        coe_value_detached: torch.Tensor,
        attention_probs_detached: torch.Tensor,
        step: Optional[int],
    ):
        base_pair = self._sample_rel_pairs_b0(base_logits_detached, sample_spec)
        rel_pair = self._sample_rel_pairs_b0(rel_logits_detached, sample_spec)
        rel_eff_pair = self._sample_rel_pairs_b0(rel_effective_detached, sample_spec)
        rel_stats = self._rel_summary_stats(rel_pair)
        rel_eff_stats = self._rel_summary_stats(rel_eff_pair)
        base_stats = self._rel_summary_stats(base_pair)
        attn_shape = self._rel_attn_shape_stats(attention_probs_detached, sample_spec, tau=int(self.log_rel_tail_tau))
        rms_base = self._rms(base_pair)
        rms_rel = self._rms(rel_pair)
        rms_rel_eff = self._rms(rel_eff_pair)
        rho = float(rms_rel_eff / max(rms_base, 1e-12)) if math.isfinite(rms_base) and math.isfinite(rms_rel_eff) else float("nan")

        z_rows = self._sample_query_rows_b0(total_logits_detached, sample_spec)
        b_rows = self._sample_query_rows_b0(base_logits_detached, sample_spec)
        r_rows = self._sample_query_rows_b0(rel_logits_detached, sample_spec)
        logits_std_vals, logits_gap_vals, logits_range_vals, corr_vals = [], [], [], []
        for qi, m in enumerate(sample_spec["query_positions"]):
            m = int(m)
            if qi < 0 or qi >= z_rows.shape[2]:
                continue
            z_valid = z_rows[:, :, qi, : m + 1]
            if z_valid.shape[-1] == 0:
                continue
            logits_std_vals.append(float(z_valid.std(dim=-1, unbiased=False).mean().item()))
            topk = torch.topk(z_valid, k=min(2, int(z_valid.shape[-1])), dim=-1).values
            if topk.shape[-1] == 1:
                gap = torch.zeros_like(topk[..., 0])
            else:
                gap = topk[..., 0] - topk[..., 1]
            logits_gap_vals.append(float(gap.mean().item()))
            logits_range_vals.append(float((z_valid.max(dim=-1).values - z_valid.mean(dim=-1)).mean().item()))

            b_valid = b_rows[:, :, qi, : m + 1]
            r_valid = r_rows[:, :, qi, : m + 1]
            for hid in range(int(b_valid.shape[1])):
                corr = self._corr_flat(b_valid[:, hid, :], r_valid[:, hid, :])
                if math.isfinite(corr):
                    corr_vals.append(float(corr))

        logits_std = float(sum(logits_std_vals) / len(logits_std_vals)) if logits_std_vals else float("nan")
        top1_gap = float(sum(logits_gap_vals) / len(logits_gap_vals)) if logits_gap_vals else float("nan")
        logits_range = float(sum(logits_range_vals) / len(logits_range_vals)) if logits_range_vals else float("nan")
        corr_base_rel = float(sum(corr_vals) / len(corr_vals)) if corr_vals else float("nan")

        bucket = self._rel_eval_buffer
        if not isinstance(bucket, dict) or bucket.get("layer_id", None) != int(layer_id):
            bucket = {
                "layer_id": int(layer_id),
                "count": 0.0,
                "sum_rms_base": 0.0,
                "sum_rms_rel": 0.0,
                "sum_rms_rel_eff": 0.0,
                "sum_rho": 0.0,
                "sum_base_abs_mean": 0.0,
                "sum_rel_mean": 0.0,
                "sum_rel_abs_mean": 0.0,
                "sum_rel_std": 0.0,
                "sum_rel_min": 0.0,
                "sum_rel_max": 0.0,
                "sum_rel_p90": 0.0,
                "sum_rel_p99": 0.0,
                "sum_rel_eff_mean": 0.0,
                "sum_rel_eff_abs_mean": 0.0,
                "sum_rel_eff_std": 0.0,
                "sum_rel_eff_min": 0.0,
                "sum_rel_eff_max": 0.0,
                "sum_rel_eff_p90": 0.0,
                "sum_rel_eff_p99": 0.0,
                "sum_ent": 0.0,
                "sum_top1": 0.0,
                "sum_tail": 0.0,
                "sum_logits_std": 0.0,
                "sum_top1_gap": 0.0,
                "sum_logits_range": 0.0,
                "sum_corr_base_rel": 0.0,
                "corr_base_rel_count": 0.0,
                "sum_coe": 0.0,
                "step": -1,
                "tail_tau": int(self.log_rel_tail_tau),
            }
        bucket["count"] += 1.0
        bucket["sum_rms_base"] += float(rms_base)
        bucket["sum_rms_rel"] += float(rms_rel)
        bucket["sum_rms_rel_eff"] += float(rms_rel_eff)
        bucket["sum_rho"] += float(rho)
        bucket["sum_base_abs_mean"] += float(base_stats["abs_mean"])
        bucket["sum_rel_mean"] += float(rel_stats["mean"])
        bucket["sum_rel_abs_mean"] += float(rel_stats["abs_mean"])
        bucket["sum_rel_std"] += float(rel_stats["std"])
        bucket["sum_rel_min"] += float(rel_stats["min"])
        bucket["sum_rel_max"] += float(rel_stats["max"])
        bucket["sum_rel_p90"] += float(rel_stats["p90"])
        bucket["sum_rel_p99"] += float(rel_stats["p99"])
        bucket["sum_rel_eff_mean"] += float(rel_eff_stats["mean"])
        bucket["sum_rel_eff_abs_mean"] += float(rel_eff_stats["abs_mean"])
        bucket["sum_rel_eff_std"] += float(rel_eff_stats["std"])
        bucket["sum_rel_eff_min"] += float(rel_eff_stats["min"])
        bucket["sum_rel_eff_max"] += float(rel_eff_stats["max"])
        bucket["sum_rel_eff_p90"] += float(rel_eff_stats["p90"])
        bucket["sum_rel_eff_p99"] += float(rel_eff_stats["p99"])
        bucket["sum_ent"] += float(attn_shape["ent"])
        bucket["sum_top1"] += float(attn_shape["top1"])
        bucket["sum_tail"] += float(attn_shape["tail"])
        bucket["sum_logits_std"] += float(logits_std)
        bucket["sum_top1_gap"] += float(top1_gap)
        bucket["sum_logits_range"] += float(logits_range)
        if math.isfinite(corr_base_rel):
            bucket["sum_corr_base_rel"] += float(corr_base_rel)
            bucket["corr_base_rel_count"] += 1.0
        bucket["sum_coe"] += float(coe_value_detached.detach().float().mean().item())
        if step is not None:
            bucket["step"] = int(step)
        self._rel_eval_buffer = bucket

    def _rel_prepare_debug(
        self,
        *,
        layer_idx: Optional[int],
        global_step,
        base_logits_tensor: Optional[torch.Tensor],
        total_logits_tensor: Optional[torch.Tensor],
        rel_logits_tensor: Optional[torch.Tensor],
        coe_value,
        attention_probs_detached: Optional[torch.Tensor],
    ):
        if (not self.log_rel_stats) or (base_logits_tensor is None) or (total_logits_tensor is None) or (rel_logits_tensor is None):
            return
        if base_logits_tensor.dim() != 4 or total_logits_tensor.dim() != 4 or rel_logits_tensor.dim() != 4:
            return
        if attention_probs_detached is None or attention_probs_detached.dim() != 4:
            return

        layer_id = self._get_rel_layer_id(layer_idx)
        train_log, train_step = self._rel_should_log_train(global_step)
        eval_log = self._rel_should_log_eval()
        if (not train_log) and (not eval_log):
            return

        sample_spec = self._build_rel_sample_spec(
            T=int(total_logits_tensor.shape[-1]),
            H=int(total_logits_tensor.shape[1]),
            device=total_logits_tensor.device,
        )
        if sample_spec is None:
            return

        coe_tensor = self._rel_to_tensor(coe_value, total_logits_tensor)
        rel_effective_detached = coe_tensor.detach() * rel_logits_tensor.detach()

        if eval_log:
            self._rel_accumulate_eval(
                layer_id=layer_id,
                sample_spec=sample_spec,
                base_logits_detached=base_logits_tensor.detach(),
                rel_logits_detached=rel_logits_tensor.detach(),
                rel_effective_detached=rel_effective_detached,
                total_logits_detached=total_logits_tensor.detach(),
                coe_value_detached=coe_tensor.detach(),
                attention_probs_detached=attention_probs_detached.detach(),
                step=self._to_int_or_none(global_step),
            )

        if train_log and total_logits_tensor.requires_grad and torch.is_grad_enabled():
            rel_ref = rel_logits_tensor
            coe_ref = coe_tensor
            hook_step = int(train_step)
            self._rel_debug_cache[layer_id] = {
                "step": int(hook_step),
                "head_count": int(sample_spec["head_idx"].numel()),
                "pair_count": int(sample_spec["m_idx"].numel()),
            }

            def _hook_fn(grad_total_logits):
                if grad_total_logits is not None:
                    self._rel_record_train_hook(
                        step=hook_step,
                        grad_total_logits=grad_total_logits,
                        rel_logits_tensor=rel_ref,
                        sample_spec=sample_spec,
                        coe_value_tensor=coe_ref,
                    )
                self._rel_debug_cache.pop(layer_id, None)

            total_logits_tensor.register_hook(_hook_fn)
        elif layer_id in self._rel_debug_cache:
            self._rel_debug_cache.pop(layer_id, None)

    @torch.no_grad()
    def _rel_pop_train_bucket(self, step: int):
        if not isinstance(self._rel_train_hook_buffer, dict):
            return None
        return self._rel_train_hook_buffer.pop(int(step), None)

    @torch.no_grad()
    def _rel_pop_eval_buffer(self):
        buf = self._rel_eval_buffer
        self._rel_eval_buffer = {}
        if isinstance(buf, dict) and float(buf.get("count", 0.0)) > 0:
            return buf
        return None

    @torch.no_grad()
    def _rel_reset_eval_buffer(self):
        self._rel_eval_buffer = {}

    def _track_attn_margin_layer(self, layer_idx: int) -> bool:
        if not self.attn_margin_enabled:
            return False
        if self.attn_margin_layers is None:
            return True
        return int(layer_idx) in self.attn_margin_layers

    @staticmethod
    @torch.no_grad()
    def _attn_margin_stat_monitor(x: torch.Tensor):
        y = x.detach().float()
        y = y.reshape(-1)
        if y.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
            }
        q = _quantiles_flat(y, qs=(0.5, 0.9, 0.99))
        return {
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()),
            "min": float(y.min().item()),
            "max": float(y.max().item()),
            "p50": float(q["p50"]),
            "p90": float(q["p90"]),
            "p99": float(q["p99"]),
        }

    @staticmethod
    @torch.no_grad()
    def _attn_margin_per_head_summary(x: torch.Tensor):
        if x.dim() != 3:
            return None, None
        y = x.detach().float().permute(1, 0, 2).reshape(x.shape[1], -1)
        if y.is_cuda:
            y = y.cpu()
        if y.numel() == 0 or y.shape[1] == 0:
            return None, None
        head_mean = y.mean(dim=-1)
        head_p90 = torch.quantile(y, 0.9, dim=-1)
        return head_mean, head_p90

    @staticmethod
    def _fmt_attn_margin_per_head(x: Optional[torch.Tensor], head_limit: int = 0) -> str:
        if x is None:
            return "[]"
        vals = x.detach().float().cpu().tolist()
        if isinstance(vals, float):
            vals = [vals]
        total = len(vals)
        if head_limit > 0 and total > head_limit:
            vals = vals[:head_limit]
            suffix = f",...(+{total - head_limit} heads)"
        else:
            suffix = ""
        body = ",".join([f"h{i}:{float(v):.6e}" for i, v in enumerate(vals)])
        return "[" + body + suffix + "]"

    def _emit_attn_margin_log(self, msg: str):
        logger_obj = getattr(self, "logger", None)
        if logger_obj is not None:
            try:
                logger_obj.info(msg)
                return
            except Exception:
                pass
        print(msg)

    @torch.no_grad()
    def _log_attn_margin_distance(
        self,
        *,
        layer_idx: Optional[int],
        masked_logits: Optional[torch.Tensor],
        global_step=None,
    ):
        if masked_logits is None:
            return
        if self.training and (not self.attn_margin_train_enabled):
            return
        if (not self.training) and (not self.attn_margin_eval_enabled):
            return
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        if not self._track_attn_margin_layer(lid):
            return

        step_val = self._to_int_or_none(global_step)
        if step_val is None and self.config is not None:
            step_val = self._to_int_or_none(getattr(self.config, "router_global_step", None))
        if step_val is None:
            self._attn_margin_local_step += 1
            step_val = int(self._attn_margin_local_step)
        if self.attn_margin_log_every > 0 and (step_val % self.attn_margin_log_every != 0):
            return

        z = masked_logits.detach().to(dtype=torch.float32)
        if z.dim() != 4 or z.shape[-1] < 2:
            return

        top2_logits = torch.topk(z, k=2, dim=-1).values
        logit_margin = top2_logits[..., 0] - top2_logits[..., 1]  # [B,H,T]

        p = torch.softmax(z, dim=-1)
        top2_probs = torch.topk(p, k=2, dim=-1).values
        prob_margin = top2_probs[..., 0] - top2_probs[..., 1]  # [B,H,T]
        entropy = -(p * (p + 1e-12).log()).sum(dim=-1)  # [B,H,T]

        metrics = [
            ("logit_margin", logit_margin),
            ("prob_margin", prob_margin),
            ("entropy", entropy),
        ]
        if self.attn_margin_log_perplexity:
            metrics.append(("perplexity", entropy.exp()))

        T = int(z.shape[2])
        for start in range(0, T, int(self.attn_margin_bin_size)):
            end = min(start + int(self.attn_margin_bin_size), T)
            if end <= start:
                continue
            bin_tag = f"{start}-{end}"
            for metric_name, metric in metrics:
                seg = metric[:, :, start:end]
                st = self._attn_margin_stat_monitor(seg)
                msg = (
                    f"[AttnMargin] layer={lid} bin={bin_tag} "
                    f"{metric_name}: mean={st['mean']:.6e} std={st['std']:.6e} "
                    f"p50={st['p50']:.6e} p90={st['p90']:.6e} p99={st['p99']:.6e} "
                    f"min={st['min']:.6e} max={st['max']:.6e} step={step_val}"
                )
                self._emit_attn_margin_log(msg)

                if self.attn_margin_log_per_head:
                    head_mean, head_p90 = self._attn_margin_per_head_summary(seg)
                    msg_head = (
                        f"[AttnMargin] layer={lid} bin={bin_tag} "
                        f"{metric_name}_per_head: "
                        f"mean={self._fmt_attn_margin_per_head(head_mean, self.attn_margin_log_head_limit)} "
                        f"p90={self._fmt_attn_margin_per_head(head_p90, self.attn_margin_log_head_limit)} "
                        f"step={step_val}"
                    )
                    self._emit_attn_margin_log(msg_head)

    @torch.no_grad()
    def reset_eval_stats(self):
        self._eval_bin_stats = {}
        self._eval_batch_step = 0
        self._eval_stats_logged_once = False

    @torch.no_grad()
    def update_stats(self, z_base: torch.Tensor, rel: torch.Tensor, layer_idx: int, rel_alpha: float = 1.0):
        if (not self._debug_enabled) or self.training:
            return
        if z_base is None or rel is None:
            return
        if not self._track_eval_layer(layer_idx):
            return
        lid = int(layer_idx)
        running = self._eval_bin_stats.get(lid)
        if running is None:
            running = RunningBinStats(
                bin_size=self.eval_stats_bin_size,
                eps=self.eval_stats_eps,
                per_head=self.eval_stats_per_head,
                stats_dtype=torch.float32,
                max_samples_per_bin=self.eval_stats_max_samples_per_bin,
            )
            self._eval_bin_stats[lid] = running
        running.update(z_base, rel, coe_for_rel=rel_alpha)

        st = self._get_layer_accum(lid)
        st["eval_bin_state"] = running.state_dict()
        st["eval_rel_alpha"] = float(rel_alpha)

        if self._is_eval_anchor_layer(lid):
            self._eval_batch_step += 1
            if self.eval_stats_log_every > 0 and (self._eval_batch_step % self.eval_stats_log_every == 0):
                self.log_stats(step=self._eval_batch_step, layer_idx=lid, force=False)

    @torch.no_grad()
    def log_stats(self, step, layer_idx: Optional[int] = None, force: bool = False):
        if not self.eval_stats_enabled:
            return
        if (not self._debug_enabled) and (not force):
            return
        if (self.eval_stats_log_every <= 0) and self.eval_stats_log_once and self._eval_stats_logged_once and (not force):
            return

        if layer_idx is None:
            layers = sorted(self._eval_bin_stats.keys())
        else:
            layers = [int(layer_idx)] if int(layer_idx) in self._eval_bin_stats else []

        for lid in layers:
            rec = self._eval_bin_stats[lid].summary()
            layer_accum = self._get_layer_accum(lid)
            alpha = float(layer_accum.get("eval_rel_alpha", self.rel_alpha))
            for bidx in sorted(rec.keys()):
                st = rec[bidx]
                t0 = int(bidx) * int(self.eval_stats_bin_size)
                t1 = t0 + int(self.eval_stats_bin_size)
                alpha_rel_over_eb = alpha * float(st.get("rel_over_eb", float("nan")))
                msg = (
                    f"[EvalStats] step={step} layer={lid} bin={t0}-{t1} "
                    f"mu_base={st['mu_base']:.6e} std_base={st['std_base']:.6e} "
                    f"mu_rel={st['mu_rel']:.6e} std_rel={st['std_rel']:.6e} "
                    f"R={st['r']:.6e} KL={st['kl']:.6e} "
                    f"alpha={alpha:.6e} rel_over_eb={st['rel_over_eb']:.6e} alpha_rel_over_eb={alpha_rel_over_eb:.6e} "
                    f"std_base_p50={st['std_base_q']['p50']:.6e} std_base_p90={st['std_base_q']['p90']:.6e} std_base_p99={st['std_base_q']['p99']:.6e} "
                    f"std_rel_p50={st['std_rel_q']['p50']:.6e} std_rel_p90={st['std_rel_q']['p90']:.6e} std_rel_p99={st['std_rel_q']['p99']:.6e} "
                    f"R_p50={st['r_q']['p50']:.6e} R_p90={st['r_q']['p90']:.6e} R_p99={st['r_q']['p99']:.6e} "
                    f"KL_p50={st['kl_q']['p50']:.6e} KL_p90={st['kl_q']['p90']:.6e} KL_p99={st['kl_q']['p99']:.6e}"
                )
                print(msg)
        if (self.eval_stats_log_every <= 0) and self.eval_stats_log_once:
            self._eval_stats_logged_once = True

    @torch.no_grad()
    def _debug_update_eval_stats(
        self, layer_idx, E_base_raw, rel,
        attn_weights=None,  # [B,H,T,T] or [B,T,H,T] depends
        router_top1=None, router_margin=None,  # 标量或向量
        max_samples_per_layer=2048
    ):
        if (not self._debug_enabled) or self.training:
            return
        if (E_base_raw is None) or (rel is None):
            return

        st = self._get_layer_accum(layer_idx)

        # abs_mean 用 sum_abs / count
        eps = 1e-12
        rel_abs_t = rel.detach().float().abs()
        eb_abs_t = E_base_raw.detach().float().abs()
        eb_abs = eb_abs_t.mean().item()
        rel_abs = rel_abs_t.mean().item()
        batch_ratio = rel_abs / max(eb_abs, eps)
        denom_floor = max(eb_abs * 1e-2, 1e-6)
        elem_ratio = (rel_abs_t / eb_abs_t.clamp_min(denom_floor)).mean().item()
        st["sum_abs_eb"]  += eb_abs
        st["sum_abs_rel"] += rel_abs
        st["sum_batch_ratio"] += batch_ratio
        st["sum_elem_ratio"] += elem_ratio
        st["count"]       += 1.0

        # extended-length bucket stats: compare ratio across sequence lengths
        seq_len = int(E_base_raw.shape[-1])
        by_len = st.get("by_seq_len")
        if by_len is None:
            by_len = {}
            st["by_seq_len"] = by_len
        bucket = by_len.get(seq_len)
        if bucket is None:
            bucket = {
                "sum_abs_rel": 0.0,
                "sum_abs_eb": 0.0,
                "sum_batch_ratio": 0.0,
                "sum_elem_ratio": 0.0,
                "count": 0.0,
            }
            by_len[seq_len] = bucket
        bucket["sum_abs_rel"] += rel_abs
        bucket["sum_abs_eb"] += eb_abs
        bucket["sum_batch_ratio"] += batch_ratio
        bucket["sum_elem_ratio"] += elem_ratio
        bucket["count"] += 1.0

        # attention entropy/top1（注意是 attention weights，不是 router）
        if attn_weights is not None:
            # 你自己确保 attn_weights 是 softmax 后的概率
            p = attn_weights.detach().float()
            # flatten over (B,H,T) per-query distribution over keys
            # 下面写法只是示意：你要按你真实维度改
            p = p.reshape(-1, p.shape[-1])  # [Q, K]
            # entropy
            ent = -(p * (p.clamp_min(1e-9).log())).sum(dim=-1)  # [Q]
            top1 = p.max(dim=-1).values                         # [Q]
            st["sum_entropy"] += ent.mean().item()
            st["sum_top1"]    += top1.mean().item()

            # reservoir sample（采一些 tail）
            if len(st["samples_attn_top1"]) < max_samples_per_layer:
                take = min(max_samples_per_layer - len(st["samples_attn_top1"]), top1.numel())
                st["samples_attn_top1"].extend(top1.flatten()[:take].cpu().tolist())

        # router tail（你已有 top1_p99/margin_p99 相关张量的话，把 per-token 值采样）
        if router_top1 is not None:
            rt = router_top1.detach().float().flatten()
            if len(st["samples_router_top1"]) < max_samples_per_layer:
                take = min(max_samples_per_layer - len(st["samples_router_top1"]), rt.numel())
                st["samples_router_top1"].extend(rt[:take].cpu().tolist())

        if router_margin is not None:
            rm = router_margin.detach().float().flatten()
            if len(st["samples_router_margin"]) < max_samples_per_layer:
                take = min(max_samples_per_layer - len(st["samples_router_margin"]), rm.numel())
                st["samples_router_margin"].extend(rm[:take].cpu().tolist())

        # rel_abs sample（同理采样）
        if len(st["samples_rel_abs"]) < max_samples_per_layer:
            st["samples_rel_abs"].append(rel_abs)
        if len(st["samples_eb_abs"]) < max_samples_per_layer:
            st["samples_eb_abs"].append(eb_abs)
        if len(st["samples_batch_ratio"]) < max_samples_per_layer:
            st["samples_batch_ratio"].append(batch_ratio)
        if len(st["samples_elem_ratio"]) < max_samples_per_layer:
            st["samples_elem_ratio"].append(elem_ratio)
    def reset_parameters(self):
        if isinstance(self.coe_for_rel, nn.Parameter):
            with torch.no_grad():
                self.coe_for_rel.fill_(self._coe_for_rel_init)

    def wavelet_rel_from_M_scale_router(self,
        q: torch.Tensor,            # [B,T,H,D]
        w: torch.Tensor,            # [B,T,H,D]
        M: torch.Tensor,            # [B,H,T,T]
        wavelet_dtt: torch.Tensor,  # [D,T,T]
        compute_dtype: torch.dtype = torch.float32,
        d_chunk: int = 8,
        layer_idx: int = None,
        rel_selection: str = "all",     # "rel1" | "rel2" | "all"
        # gates (推荐 token-wise): [B,T,H,S]，也可广播成 [1,1,H,S] / [B,1,H,S]
        gate1: torch.Tensor = None,     # for rel1
        gate2: torch.Tensor = None,     # for rel2
        # fallback coe（如果你还想保留 head-wise 标量）: [H] or [H,1,1]
        scale_wise_analyzer=None,
        E_base_raw: torch.Tensor = None,  # [B,H,T,T]
        config=None,
        global_step=None,
    ) -> torch.Tensor:
        """
        Scale-wise routed:
        rel1 = sum_s (gate1 * q_s) P_s
        rel2 = sum_s (gate2 * qcorr_s) P_s
        rel  = rel1 - rel2   (or selection)
        Return: rel [B,H,T,T] (NO scale, NO mask)
        """
        self._rel_last_raw_logits = None
        self._rel_last_coe_value = None
        hier = bool(getattr(self.config, "hierarchical_gate_use", False))
        global_lambda = float(getattr(self.config, "global_lambda", 0.5)) 
        B, T, H, D = q.shape
        assert wavelet_dtt.shape[0] == D
        assert D % d_chunk == 0
        S = D // d_chunk

        q0 = q.to(compute_dtype)
        w0 = w.to(compute_dtype)
        P  = wavelet_dtt.to(compute_dtype)

        # P_s: [S,T,T]  (同组共享scale -> 压缩到scale-group)
        # Current experiment default: always use averaged scale-group path.
        # Keep shift-separate branch disabled to avoid accidental config toggles.
        shift_sep_use = False
        if shift_sep_use:
            P_s = P.view(S, d_chunk, T, T)
            q_s = q0.view(B, T, H, S, d_chunk)
        else:
            P_s = P.view(S, d_chunk, T, T).mean(dim=1)
            q_s = q0.view(B, T, H, S, d_chunk).sum(dim=-1)

        # q_s: [B,T,H,S]  (组内求和；也可以改成 mean，看你定义)
        #
        def _to_int_or_none(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                if x.numel() != 1:
                    return None
                x = x.detach().item()
            try:
                return int(x)
            except Exception:
                return None

        do_router_norm = bool(self.router_norm_cfg.enable)
        step_val = None
        emit_header = False
        do_router_norm_log = False
        if do_router_norm:
            step_val = _to_int_or_none(global_step)
            if step_val is None and config is not None:
                step_val = _to_int_or_none(getattr(config, "router_global_step", None))
            if step_val is None:
                self._router_norm_local_step += 1
                step_val = int(self._router_norm_local_step)
            emit_header = True
            do_router_norm_log = bool(
                self.router_norm_logger is not None and self.router_norm_logger.should_log(step_val)
            )

        @torch.no_grad()
        def _log_non_finite(tag: str, x: Optional[torch.Tensor]):
            if (not do_router_norm_log) or (x is None):
                return
            xf = x.detach().float()
            numel = int(xf.numel())
            if numel == 0:
                return
            finite = torch.isfinite(xf)
            n_finite = int(finite.sum().item())
            if n_finite == numel:
                return
            n_nan = int(torch.isnan(xf).sum().item())
            n_inf = int(torch.isinf(xf).sum().item())
            layer_repr = "NA" if layer_idx is None else str(int(layer_idx))
            print(
                f"[RouterNorm][NonFinite] step={step_val} layer={layer_repr} tag={tag} "
                f"finite={n_finite}/{numel} nan={n_nan} inf={n_inf}"
            )

        # gate 默认：全 1（不路由）
        if hier:
            # local logits: [B,T,H,S]  (head_dim -> S)
            local_logits1 = self.local_router1(q0)

            # global logits: reshape q0 -> [B,T,H*D] == hidden_size -> [B,T,S]
            q_global = q0.reshape(B, T, H * D)
            # 强烈建议加一个 assert，防止未来改了模型结构
            assert q_global.shape[-1] == self.hidden_size, f"H*D={H*D} != hidden_size={self.hidden_size}"
            global_logits1 = self.global_router1(q_global).unsqueeze(2)  # [B,T,1,S]

            mix_logits1 = local_logits1 + global_lambda * global_logits1
            mix_logits1 = mix_logits1
            gate1 = torch.softmax(mix_logits1, dim=-1)  # [B,T,H,S]
        else:
            if gate1 is None:
                gate1 = 1.0
                gate2 = 1.0
            if gate1 is not None and gate2 is None:
                gate2 = gate1
            # rel1: [B,H,T,T]
        rel1 = None
        if rel_selection in ("rel1", "all"):
            if do_router_norm:
                rel1 = apply_router_norm_mode(
                    q_like=q_s,
                    gate=gate1,
                    p_s=P_s,
                    router_norm=self.router_norm,
                    cfg=self.router_norm_cfg,
                    shift_sep_use=bool(shift_sep_use),
                    logger=self.router_norm_logger,
                    step=step_val,
                    layer_idx=layer_idx,
                    tensor_name="q_s",
                    gate_name="gate1",
                    emit_header=emit_header,
                )
                emit_header = False
            else:
                if not shift_sep_use:
                    rel1 = torch.einsum("b t h s, s t n -> b h t n", gate1 * q_s, P_s)
                else:
                    gate1_c = gate1.unsqueeze(-1)
                    rel1 = torch.einsum("b t h s c, s c t n -> b h t n", gate1_c * q_s, P_s)

            if self.rel1_coe is not None:
                # rel1_coe: [H] or [H,1,1] -> broadcast to [B,H,T,T]
                rel1 = rel1 * self.rel1_coe

        # rel2: [B,H,T,T]
        rel2 = None
        q_corr = None
        if rel_selection in ("rel2", "all"):
            # q_corr: [B,T,H,D] = M W
            if not shift_sep_use:
                m_used = M.to(compute_dtype)
                _log_non_finite("M", m_used)
                _log_non_finite("w0", w0)
                q_corr = torch.einsum("b h t j, b j h d -> b t h d", m_used, w0)
                _log_non_finite("q_corr", q_corr)
                if hier:
                    local_logits2 = self.local_router2(q_corr)  # [B,T,H,S]

                    qcorr_global = q_corr.reshape(B, T, H * D)
                    assert qcorr_global.shape[-1] == self.hidden_size, f"H*D={H*D} != hidden_size={self.hidden_size}"
                    global_logits2 = self.global_router2(qcorr_global).unsqueeze(2)  # [B,T,1,S]

                    mix_logits2 = local_logits2 + global_lambda * global_logits2
                    mix_logits2 = mix_logits2
                    gate2 = torch.softmax(mix_logits2, dim=-1)  # [B,T,H,S]
                qcorr_s = q_corr.view(B, T, H, S, d_chunk).sum(dim=-1)
                _log_non_finite("qcorr_s", qcorr_s)
                if do_router_norm:
                    rel2 = apply_router_norm_mode(
                        q_like=qcorr_s,
                        gate=gate2,
                        p_s=P_s,
                        router_norm=self.router_norm,
                        cfg=self.router_norm_cfg,
                        shift_sep_use=False,
                        logger=self.router_norm_logger,
                        step=step_val,
                        layer_idx=layer_idx,
                        tensor_name="qcorr_s",
                        gate_name="gate2",
                        emit_header=emit_header,
                    )
                    emit_header = False
                else:
                    rel2 = torch.einsum("b t h s, s t n -> b h t n", gate2 * qcorr_s, P_s)
            else:
                m_used = M.to(compute_dtype)
                _log_non_finite("M", m_used)
                _log_non_finite("w0", w0)
                q_corr = torch.einsum("b h t j, b j h d -> b t h d", m_used, w0)
                _log_non_finite("q_corr", q_corr)
                qcorr_s = q_corr.view(B, T, H, S, d_chunk)
                _log_non_finite("qcorr_s", qcorr_s)
                if do_router_norm:
                    rel2 = apply_router_norm_mode(
                        q_like=qcorr_s,
                        gate=gate2,
                        p_s=P_s,
                        router_norm=self.router_norm,
                        cfg=self.router_norm_cfg,
                        shift_sep_use=True,
                        logger=self.router_norm_logger,
                        step=step_val,
                        layer_idx=layer_idx,
                        tensor_name="qcorr_s",
                        gate_name="gate2",
                        emit_header=emit_header,
                    )
                    emit_header = False
                else:
                    gate2_c = gate2.unsqueeze(-1)
                    rel2 = torch.einsum("b t h s c, s c t n -> b h t n", gate2_c * qcorr_s, P_s)

            if self.wavelet_coe is not None:
                rel2 = rel2 * self.wavelet_coe

        # combine
        if rel_selection == "rel1":
            rel_raw = rel1
            coe_for_rel = 1.0
            rel = rel_raw
        elif rel_selection == "rel2":
            rel_raw = -rel2
            coe_for_rel = 1.0
            rel = rel_raw
        elif rel_selection == "all":
            rel_raw = rel1 - rel2
            coe_for_rel = self.coe_for_rel
            # if isinstance(coe_for_rel, torch.Tensor) and coe_for_rel.requires_grad:
            #     coe_for_rel = torch.sigmoid(coe_for_rel)
            coe_for_rel = coe_for_rel * float(getattr(config, "rel_zoom_in_coe", 1.0))
            rel = coe_for_rel * rel_raw
        else:
            raise ValueError(f"Unknown rel_selection={rel_selection}")
        self._rel_last_raw_logits = rel_raw
        self._rel_last_coe_value = coe_for_rel

        # analyzer update: only when q_corr is available (or pass None and handle it in analyzer)
        if scale_wise_analyzer is not None and (E_base_raw is not None) and (q_corr is not None):
            # 这里传 gate/coe 你想记录什么都行；先沿用 rel2_coe
            scale_wise_analyzer.update(layer_idx, E_base_raw, q, q_corr, wavelet_dtt, coe=rel2_coe)
        data_collection_in_layerwise = bool(getattr(self.config, "data_collection_in_layerwise", False))
        if data_collection_in_layerwise:
            if layer_idx is not None:
                # build save dir
                save_root = getattr(self.config, "save_root", None)
                if save_root is None:
                    raise ValueError("self.config.save_root is None but rel_record dumping is enabled.")

                dump_dir = os.path.join(save_root, "rel_record")
                os.makedirs(dump_dir, exist_ok=True)

                # create a unique filename to avoid overwrite if forward called multiple times
                # (e.g., gradient accumulation / multiple microbatches)
                rec_i = int(getattr(self, "_rel_record_i", 0))
                setattr(self, "_rel_record_i", rec_i + 1)

                gpu_id = os.environ.get("LOCAL_RANK")
                if gpu_id is None:
                    if torch.cuda.is_available():
                        gpu_id = str(torch.cuda.current_device())
                    elif dist.is_available() and dist.is_initialized():
                        gpu_id = str(dist.get_rank())
                    else:
                        gpu_id = "cpu"
                fname = f"layer{int(layer_idx):02d}_T{T}_H{H}_gpu{gpu_id}.pt"
                fpath = os.path.join(dump_dir, fname)

                # move to cpu to reduce GPU memory pressure
                payload = {
                    "meta": {
                        "layer_idx": int(layer_idx),
                        "B": int(B), "T": int(T), "H": int(H), "D": int(D),
                        "S": int(S), "d_chunk": int(d_chunk),
                        "shift_sep_use": bool(shift_sep_use),
                        "hierarchical_gate_use": bool(hier),
                        "rel_selection": str(rel_selection),
                    },
                    # tensors: save full batch/token matrices for later analysis
                    # "rel1": (rel1.detach().to("cpu") if rel1 is not None else None),
                    # "rel2": (rel2.detach().to("cpu") if rel2 is not None else None),
                    # "rel":  (rel.detach().to("cpu")  if rel  is not None else None),
                    # optional: gates are extremely informative for your “mixture kernel” visualization
                    "gate1": (gate1.detach().to("cpu") if isinstance(gate1, torch.Tensor) else None),
                    "gate2": (gate2.detach().to("cpu") if isinstance(gate2, torch.Tensor) else None),
                }

                # stats + safety checks (monitor logging)
                # stats = {}
                # stats.update(_tensor_stats(rel1, "rel1"))
                # stats.update(_tensor_stats(rel2, "rel2"))
                # stats.update(_tensor_stats(rel,  "rel"))
                # if isinstance(gate1, torch.Tensor):
                #     stats.update(_tensor_stats(gate1, "gate1"))
                # if isinstance(gate2, torch.Tensor):
                #     stats.update(_tensor_stats(gate2, "gate2"))
                # stats["save_path"] = fpath
                # _log_stats(stats, prefix=f"[rel_record] layer={layer_idx}")

                # save
                torch.save(payload, fpath)

                # exit when layer_idx==1 (as requested)
                if int(layer_idx) == 11:
                    os._exit(0)
        return rel
    
    def path_attention_with_wavelet_QH(self,
        q, k, v, w, beta,
        wavelet_dtt,
        use_wavelet_fused_H: bool = False,
        d_chunk: int = 8,
        compute_dtype: torch.dtype = torch.float32,
        analyzer=None,
        # scale_wise_analyzer=None,
        layer_idx=None,
        ablate=None,
        rel_selection = None,
        router1=None,
        router2=None,
        config=None,
        global_step=None,
        router_log_every=None,
    ):
        # === (A) head 对齐：先沿用你现在的 Hw 对齐（但见下文我建议改成 Hq 对齐） ===
        Hw = w.shape[2]
        q = _match_heads(q, Hw)
        k = _match_heads(k, Hw)
        v = _match_heads(v, Hw)
        if beta.dim() == 3:  # [B,T,H]
            beta = _match_heads(beta.unsqueeze(-1), Hw)[..., 0]
        else:
            beta = beta  # assume already [B,T,Hw]

        B, T, H, d = q.shape
        scale = d ** -0.5
        future = _future_mask(T, q.device)  # [1,1,T,T]

        # --- baseline raw logits and M_base ---
        E_base_raw, M_base, strict_WK, A = path_ut_base_raw(
            q, k, w, beta, compute_dtype=compute_dtype
        )
        # --- pick M_used for defining QH in wavelet branch ---
        if use_wavelet_fused_H:
            M_used = path_ut_M_wave_fused(q, w, beta, A, wavelet_dtt, d_chunk=d_chunk, compute_dtype=compute_dtype)
        else:
            M_used = M_base
        # dump_last_query_per_dim(
        #     save_path=f"./D_keep_data/hotpot_qa_2048L_mix_PA_pretrain_WR_layer{layer_idx:02d}",
        #     q=q, k=k, w=w,
        #     M=M_used,                     # 用你想分析的那份 M（M_base 或 M_wave）
        #     wavelet_dtt=wavelet_dtt,
        #     layer_idx=layer_idx,           # 视显存/速度调
        #     save_dtype=torch.float32,
        #     compute_corr_row=True,
        #     router1=router1,
        #     router2=router2,
        # )
        
        # if layer_idx == 11:
        #     os._exit(0)
        # --- wavelet rel term ---

        rel_alpha = float(getattr(config, "rel_alpha", getattr(config, "attn_rel_alpha", self.rel_alpha)))
        rel_logits_raw = None
        coe_layer = None
        if wavelet_dtt is not None and self._rel_layer_enabled(layer_idx, config=config):
            if router1 is not None and router2 is not None:
                rel = self.wavelet_rel_from_M_scale_router(q, w, M_used, wavelet_dtt, compute_dtype=compute_dtype, d_chunk=d_chunk, layer_idx=layer_idx,
                                                    rel_selection=rel_selection, gate1=router1, gate2=router2,config=config, global_step=global_step)
            else:
                raise ValueError("router1 and router2 must be provided when wavelet_dtt is not None")
            # optional ablation on rel
            if ablate is not None and layer_idx in ablate:
                for h in ablate[layer_idx]:
                    rel[:, h].zero_()
            rel_logits_raw = self._rel_last_raw_logits if torch.is_tensor(self._rel_last_raw_logits) else rel
            coe_layer = self._rel_last_coe_value
            if coe_layer is None:
                coe_layer = 1.0
            coe_layer = rel_alpha * coe_layer
            # final logits: z = base + alpha * rel
            E_wav_raw = E_base_raw + rel_alpha * rel
            
            def _to_int_or_none(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    if x.numel() != 1:
                        return None
                    x = x.detach().item()
                try:
                    return int(x)
                except Exception:
                    return None

            step_val = global_step
            if step_val is None and config is not None:
                step_val = getattr(config, "router_global_step", None)
            step_val = _to_int_or_none(step_val)

            log_every = router_log_every
            if log_every is None and config is not None:
                log_every = getattr(config, "router_log_every", 500)
            log_every = _to_int_or_none(log_every)

            should_log = (
                self.training
                and step_val is not None
                and log_every is not None
                and log_every > 0
                and step_val >= 0
                and (step_val % log_every == 0)
            )
            if should_log:
                with torch.no_grad():
                    eps = 1e-12
                    rel_abs = rel.detach().float().abs()
                    base_abs = E_base_raw.detach().float().abs()
                    rel_abs_mean = rel_abs.mean().item()
                    base_abs_mean = base_abs.mean().item()
                    # More stable "average magnitude ratio": mean(|rel|) / mean(|E_base_raw|).
                    abs_ratio_mean = rel_abs_mean / max(base_abs_mean, eps)
                    # Keep an element-wise ratio monitor with a practical denominator floor.
                    denom_floor = max(base_abs_mean * 1e-2, 1e-6)
                    abs_ratio_elem_mean = (rel_abs / base_abs.clamp_min(denom_floor)).mean().item()

                    p = rel_abs.reshape(-1)
                    q_prob = base_abs.reshape(-1)
                    p = p / p.sum().clamp_min(eps)
                    q_prob = q_prob / q_prob.sum().clamp_min(eps)
                    m = 0.5 * (p + q_prob)

                    p_log = p.clamp_min(eps).log()
                    q_log = q_prob.clamp_min(eps).log()
                    m_log = m.clamp_min(eps).log()
                    js_div = 0.5 * ((p * (p_log - m_log)).sum() + (q_prob * (q_log - m_log)).sum())
                    js_sim = 1.0 - (js_div / math.log(2.0)).item()
                    js_sim = max(0.0, min(1.0, js_sim))

                msg = (
                    f"[wavelet rel stats] layer={layer_idx} step={step_val} "
                    f"abs_ratio_mean={abs_ratio_mean:.6e} "
                    f"abs_ratio_elem_mean={abs_ratio_elem_mean:.6e} "
                    f"rel_abs_mean={rel_abs_mean:.6e} base_abs_mean={base_abs_mean:.6e} "
                    f"js_sim={js_sim:.6f}"
                )
                logger_obj = getattr(self, "logger", None)
                if logger_obj is not None:
                    try:
                        logger_obj.info(msg)
                    except Exception:
                        print(msg)
                else:
                    print(msg)
        else:
            rel = None
            E_wav_raw = E_base_raw

        # Persist attention scores for later analysis when enabled
        # path = Path(config.model_name_or_path)
        # dump_dir = f"./attention_score_record/L{config.block_size}_{config.dataset_name}_{path.parent.name}_{path.name}"
        # _ensure_dir(dump_dir)
        # tag = f"layer{layer_idx:02d}" if layer_idx is not None else "layer_unknown"
        # fname = f"{tag}_{int(time.time() * 1000)}.pt"
        # payload = {
        #     "layer_idx": layer_idx,
        #     "has_wavelet": wavelet_dtt is not None,
        #     "E_wav_raw": E_wav_raw.detach().to(torch.float32).cpu(),
        # }
        # if wavelet_dtt is not None:
        #     payload["E_base_raw"] = E_base_raw.detach().to(torch.float32).cpu()
        #     payload["rel"] = rel.detach().to(torch.float32).cpu() if rel is not None else None
        # _save_block(dump_dir, fname, payload)
        # if layer_idx == 11:
        #     os._exit(0)
        # Persist attention scores for later analysis when enabled

        E_wav = E_wav_raw * scale
        wave_fill = causal_mask_fill_value(E_wav.dtype)
        E_wav = E_wav.masked_fill(future, wave_fill)
        self._log_attn_margin_distance(
            layer_idx=layer_idx,
            masked_logits=E_wav,
            global_step=global_step,
        )
        P_wav = torch.softmax(E_wav, dim=-1)
        self._rel_prepare_debug(
            layer_idx=layer_idx,
            global_step=global_step,
            base_logits_tensor=E_base_raw,
            total_logits_tensor=E_wav_raw,
            rel_logits_tensor=rel_logits_raw,
            coe_value=coe_layer,
            attention_probs_detached=P_wav,
        )
        self._debug_update_eval_stats(layer_idx, E_base_raw, rel, attn_weights=P_wav)
        self.update_stats(E_base_raw, rel, layer_idx, rel_alpha=rel_alpha)
        out_wav = torch.einsum("b h i j, b j h d -> b i h d", P_wav, v.to(compute_dtype))

        if analyzer is not None and rel is not None:
            w0 = w.to(compute_dtype)
            deltaQ = torch.einsum("b h i j, b j h d -> b i h d", M_used, w0)
            analyzer['layer_attention_analyzer'].update(layer_idx, E_base_raw, rel, P_base, P_wav, q, deltaQ)

        pwav_logger = analyzer.get("pwav_mean_logger") if isinstance(analyzer, dict) else None
        if pwav_logger is not None:
            pwav_logger.update(layer_idx, out_wav)
        return out_wav
    def _log_head_vector(self, name: str, vec: torch.Tensor, fmt: str = "{:.4f}", topk: Optional[int] = None):
        try:
            v = vec.detach().float().cpu()
            vals = v.tolist()
            if isinstance(vals, float):  # 标量容错
                msg = fmt.format(vals)
            else:
                if topk is not None and len(vals) > topk:
                    msg = " ".join(f"h{i}:{fmt.format(vals[i])}" for i in range(topk))
                    msg += f" ... (+{len(vals)-topk} heads)"
                else:
                    msg = " ".join(f"h{i}:{fmt.format(x)}" for i, x in enumerate(vals))
            self.logger.info(f"{name}: {msg}")
        except Exception as e:
            print(f"[PaTHAttention][log error] {name}: {e}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        wavelet_decay_table: Optional[torch.Tensor] = None,  # [B,T,H,] or None
        geom_p = 0,
        analyzer=None,
        scale_wise_analyzer=None,
        router_analyzer=None,
        input_ids=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if use_cache:
            assert past_key_values is not None, "past_key_values must be provided when use_cache is True"
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        q = self.q_proj(hidden_states)           # [B,T,Hq*d]
        k = self.k_proj(hidden_states)           # [B,T,H*d]
        v = self.v_proj(hidden_states)           # [B,T,H*d]
        w = self.w_proj(hidden_states)           # [B,T,H*R*d]
        global_step = kwargs.get("global_step", getattr(self.config, "router_global_step", None))
        router1, router2 = None, None
        record_router1, record_router2 = None, None
        rel_use = self._rel_layer_enabled(self.layer_idx, self.config)
        router_active = bool(self.config.wavelet_router) and bool(rel_use)
        if router_active:
            B = w.size(0)
            S=8
            H= self.num_heads
            T = w.size(1)
# ---- router ----
            router_gate_use = getattr(self.config, "router_gate_use", False)
            if router_gate_use:
                z1 = self.low_rank_map1(hidden_states)          # [B,T,32]
                logits1 = self.router1_head(z1).reshape(B,T,H,S)   # [B,T,H,S]
                router1_b = torch.softmax(logits1 / self.tau_router, dim=-1)

                gate_logits1 = self.router1_gate_head(z1)       # [B,T,H]
                lam1 = torch.sigmoid(gate_logits1 / self.t_gate)

                z2 = self.low_rank_map2(hidden_states)          # [B,T,32]
                logits2 = self.router2_head(z2).reshape(B,T,H,S)
                router2_b = torch.softmax(logits2 / self.tau_router, dim=-1)

                gate_logits2 = self.router2_gate_head(z2)       # [B,T,H]
                lam2 = torch.sigmoid(gate_logits2 / self.t_gate)

                router1 = lam1.unsqueeze(-1) * router1_b
                router2 = lam2.unsqueeze(-1) * router2_b
            else:
                if getattr(self.config, "hierarchical_gate_use", False):
                    pass
                else:
                    router1_logits = self.router1(hidden_states)          # [B,T,H*S]
                    router1_logits = router1_logits.view(B, T, H, S)      # [B,T,H,S]
                    # jitter hyperparams
                    jitter_std = getattr(self.config, "router_jitter_std", 0.0)   # e.g. 0.01
                    jitter_apply_in_eval = getattr(self.config, "router_jitter_apply_in_eval", False)

                    # 进度感知：前 30%/后 70% 使用不同 std（如未提供则退回默认）
                    # 优先用外部传入的 global_step（通常是优化步）；否则用 config 预设；再否则用本地计数，
                    # 本地计数按前向 micro-step 计数，并除以 grad_accum_steps 以对齐优化步。
                    global_step_ext = kwargs.get("global_step", getattr(self.config, "router_global_step", None))
                    grad_accum = kwargs.get(
                        "grad_accum_steps",
                        getattr(self.config, "router_grad_accum_steps", getattr(self.config, "gradient_accumulation_steps", 1)),
                    )
                    if global_step_ext is None:
                        micro_step = getattr(self, "router_local_step", 0)
                        global_step = micro_step // max(1, int(grad_accum))
                        if self.training and torch.is_grad_enabled():
                            self.router_local_step = micro_step + 1
                        max_steps = kwargs.get("max_steps", getattr(self.config, "router_max_steps", 15900))
                    else:
                        global_step = global_step_ext
                        max_steps = kwargs.get("max_steps", getattr(self.config, "router_max_steps", 15900))
                    jitter_std_early = getattr(self.config, "router_jitter_std_early", jitter_std)
                    jitter_std_late = getattr(self.config, "router_jitter_std_late", jitter_std)
                    if (global_step is not None) and (max_steps not in (None, 0)):
                        pct = float(global_step) / float(max_steps)
                        jitter_std = jitter_std_early if pct < 0.3 else jitter_std_late

                    # After step 10000, anneal jitter_std from 1 -> 0 over 1000 steps.
                    jitter_std_end_ratio = getattr(self.config, "jitter_std_end_ratio", 0.0)
                    jitter_anneal_span = getattr(self.config, "jitter_anneal_span", 1000)
                    jitter_anneal_start_step = getattr(self.config, "jitter_anneal_start_step", 10000)
                    if jitter_anneal_span > 0:
                        if global_step is not None and global_step >= jitter_anneal_start_step:
                            slope = (jitter_std_end_ratio - 1.0) / jitter_anneal_span
                            coeff = 1.0 + slope * (global_step - jitter_anneal_start_step)
                            jitter_std = max(jitter_std_end_ratio, min(coeff, 1.0)) * jitter_std

                    # 日志监控：每 log_every 步打印一次当前 jitter_std
                    log_every = getattr(self.config, "router_jitter_log_every", 1000)
                    log_noise_stats = (
                        log_every > 0
                        and global_step is not None
                        and max_steps not in (None, 0)
                        and (global_step % log_every == 0)
                    )

                    def _add_gaussian_jitter(logits: torch.Tensor, std: float, router_name=None) -> torch.Tensor:
                        """
                        logits: [B, T, H, S]
                        train: return logits + noise
                        eval : NEVER add noise to logits; only compute & log stats, then return logits
                        """
                        if std <= 0:
                            return logits

                        # ---- config ----
                        noise_adapt_style = getattr(self.config, "noise_adapt_style", "logit_std")

                        sigma_max_cfg = getattr(self.config, "router_jitter_sigma_max", None)
                        sigma_max = float("inf") if sigma_max_cfg is None else float(sigma_max_cfg)

                        sigma_min_cfg = getattr(self.config, "router_jitter_sigma_min", None)
                        sigma_min = 0.0 if sigma_min_cfg is None else float(sigma_min_cfg)

                        # logging switches (train/eval)
                        log_router_stats_train = bool(getattr(self.config, "log_train_router_stats", True))
                        log_router_stats_eval  = bool(getattr(self.config, "log_eval_router_stats", True))
                        log_every = int(getattr(self.config, "router_log_every", 500))

                        logits_det = logits.detach()
                        B, T, H, S = logits_det.shape
                        if S < 2:
                            return logits

                        # =========================
                        # ---- compute sigma_raw / sigma_eff ----
                        # =========================
                        sigma_raw = None
                        sigma_eff = None

                        if noise_adapt_style in ("fliprate_head", "fliprate_token"):
                            # std is interpreted as target flip prob rho0
                            rho0 = float(std)
                            rho0 = min(max(rho0, 1e-6), 0.499999)

                            # margin = top1 - top2  (per token, per head)
                            top2 = torch.topk(logits_det, k=2, dim=-1).values  # [B,T,H,2]
                            margin = (top2[..., 0] - top2[..., 1]).clamp_min(1e-6)  # [B,T,H]

                            # denom from Gaussian flip approximation
                            normal = torch.distributions.Normal(
                                loc=logits_det.new_tensor(0.0),
                                scale=logits_det.new_tensor(1.0),
                            )
                            z = normal.icdf(logits_det.new_tensor(rho0)).abs().clamp_min(1e-6)
                            denom = (math.sqrt(2.0) * z)  # scalar

                            if noise_adapt_style == "fliprate_head":
                                # head-wise: reduce (B,T) -> head scalar
                                margin_bt = margin.reshape(-1, H)                 # [BT,H]
                                margin_head = margin_bt.median(dim=0).values      # [H]
                                sigma_head_raw = (margin_head / denom)            # [H]
                                sigma_head_eff = sigma_head_raw.clamp(min=sigma_min, max=sigma_max)

                                sigma_raw = sigma_head_raw.view(1, 1, H, 1)       # broadcast [B,T,H,1]
                                sigma_eff = sigma_head_eff.view(1, 1, H, 1)

                            else:
                                # token-wise: sigma depends on (B,T,H)
                                sigma_tok_raw = (margin / denom)                  # [B,T,H]
                                sigma_tok_eff = sigma_tok_raw.clamp(min=sigma_min, max=sigma_max)

                                sigma_raw = sigma_tok_raw.unsqueeze(-1)           # [B,T,H,1]
                                sigma_eff = sigma_tok_eff.unsqueeze(-1)

                        elif noise_adapt_style == "logit_std":
                            scale = logits_det.std(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,T,H,1]
                            sigma_raw = float(std) * scale
                            sigma_eff = sigma_raw.clamp(min=sigma_min, max=sigma_max)

                        elif noise_adapt_style == "const":
                            sigma_raw = logits.new_full(logits.shape[:-1] + (1,), float(std))
                            sigma_eff = sigma_raw.clamp(min=sigma_min, max=sigma_max)

                        else:
                            raise ValueError(f"Unknown noise_adapt_style: {noise_adapt_style}")

                        # =========================
                        # ---- (stat monitor) log in train only ----
                        # =========================
                        def _should_log() -> bool:
                            try:
                                return (int(global_step) % log_every == 0 and self.training)
                            except Exception:
                                return True

                        def _fmt_vec(x: torch.Tensor, nd: int = 4) -> str:
                            x = x.detach().float().cpu().tolist()
                            return "[" + ",".join([f"{v:.{nd}f}" for v in x]) + "]"

                        do_log = _should_log()
                        want_log = self.training and log_router_stats_train

                        if do_log and want_log:
                            try:
                                # reshape to [BT,H]
                                raw_bt = sigma_raw.detach().reshape(-1, H)
                                eff_bt = sigma_eff.detach().reshape(-1, H)

                                q = torch.tensor([0.5, 0.9, 0.95], device=raw_bt.device)
                                raw_q = torch.quantile(raw_bt, q, dim=0)  # [3,H]
                                eff_q = torch.quantile(eff_bt, q, dim=0)  # [3,H]

                                if math.isfinite(sigma_max):
                                    clip_rate_raw = (raw_bt > sigma_max).float().mean(dim=0)  # [H]
                                else:
                                    clip_rate_raw = torch.zeros((H,), device=logits.device)

                                # compression ratio E[eff/raw]
                                comp = (eff_bt / raw_bt.clamp_min(1e-12)).mean(dim=0)  # [H]

                                msg = (
                                    f"router_name={router_name} "
                                    f"[router {'train' if self.training else 'eval'} stats] "
                                    f"layer={self.layer_idx} step={global_step}/{max_steps} style={noise_adapt_style} "
                                    f"rho_or_std={std} sigmax={sigma_max_cfg} sigmin={sigma_min_cfg} "
                                    f"raw_p50={_fmt_vec(raw_q[0], 6)} raw_p90={_fmt_vec(raw_q[1], 6)} raw_p95={_fmt_vec(raw_q[2], 6)} "
                                    f"eff_p50={_fmt_vec(eff_q[0], 6)} eff_p90={_fmt_vec(eff_q[1], 6)} eff_p95={_fmt_vec(eff_q[2], 6)} "
                                    f"clip_rate_raw={_fmt_vec(clip_rate_raw, 4)} comp_mean={_fmt_vec(comp, 4)}"
                                )

                                logger_obj = getattr(self, "logger", None)
                                if logger_obj is not None:
                                    try:
                                        logger_obj.info(msg)
                                    except Exception:
                                        print(msg)
                                else:
                                    print(msg)
                            except Exception:
                                pass

                        # =========================
                        # Eval: never inject
                        # =========================
                        if not self.training:
                            return logits

                        # =========================
                        # Train: inject
                        # =========================
                        noise = torch.randn_like(logits) * sigma_eff  # [B,T,H,S]
                        return logits + noise
                    tau_change_step = getattr(self.config, "tau_change_step", 0)
                    tau_change = getattr(self.config, "tau_change", 2.0)
                    log_every = int(getattr(self.config, "router_log_every", 500))
                    if tau_change_step > 0 and global_step == tau_change_step:
                        self.tau = tau_change

                    router1_logits = _add_gaussian_jitter(router1_logits, jitter_std, router_name="router1")
                    router1 = torch.softmax(router1_logits / self.tau, dim=-1)  # [B,T,H,S]

                    if self.router2 is None:
                        router2 = None
                    else:
                        router2_logits = self.router2(hidden_states)       # [B,T,H*S]
                        router2_logits = router2_logits.view(B, T, H, S)   # [B,T,H,S]
                        router2_logits = _add_gaussian_jitter(router2_logits, jitter_std, router_name="router2")
                        router2 = torch.softmax(router2_logits / self.tau, dim=-1)  # [B,T,H,S]
                    

                    def _should_log() -> bool:
                        try:
                            return (int(global_step) % log_every == 0 and self.training)
                        except Exception:
                            return True
                    do_log = _should_log()
                    if do_log:
                        if router2 is not None:
                            _router_gate_stats(
                                router_name="router2",
                                logits=router2_logits,   # softmax 前
                                gate=router2,            # softmax 后
                                tau=float(self.tau),
                                coe_for_rel = self.coe_for_rel,
                                global_step=int(global_step),
                                log_every=int(log_every),
                                layer_idx=int(self.layer_idx),
                                logger_obj=getattr(self, "logger", None),
                            )
                        if router1 is not None:
                            _router_gate_stats(
                                router_name="router1",
                                logits=router1_logits,   # softmax 前
                                gate=router1,            # softmax 后
                                tau=float(self.tau),
                                coe_for_rel = self.coe_for_rel,
                                global_step=int(global_step),
                                log_every=int(log_every),
                                layer_idx=int(self.layer_idx),
                                logger_obj=getattr(self, "logger", None),
                            )                        

                    data_collection_style = getattr(self.config, "router_data_collection_style", None)
                    if data_collection_style == 'logit':
                        record_router1 = router1_logits
                        record_router2 = router2_logits
                    elif data_collection_style == 'prob' or data_collection_style is None:
                        record_router1 = router1
                        record_router2 = router2
                    else:
                        raise ValueError(f"Unknown data_collection_style: {data_collection_style}")
                    # if self.layer_idx==11:
                    #     os._exit(0)
            # ---- end router ----
                if analyzer is not None:
                    checkpoint = Path(self.config.model_name_or_path).name
                    proj_name = Path(self.config.model_name_or_path).parent.name
                    analyzer['router_analyzer'].update(self.layer_idx, router1, router2)
                    analyzer['token_scale_dumper'].update(step=int(checkpoint.split('-')[-1]), layer_idx=self.layer_idx, input_ids=input_ids, gate1=router1, gate2=router2)
            
        beta_logits = self.bt_proj(hidden_states)  # [B,T,H*R]
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None

        q, k = self.maybe_q_norm(q), self.maybe_k_norm(k)
        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert not (cu_seqlens is not None and attention_mask is not None), (
            "cu_seqlens should not be provided when attention_mask is not None"
        )
        # ========= 训练路径（mask=None）=========
        if attention_mask is None:
            assert use_cache is False, "use_cache should be False in training"

            # w 分支卷积
            if self.use_w_shortconv:
                w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

            # 整理到多头格式
            q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_dim)                     # HQ
            k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_dim)                     # H
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_dim)                     # H
            W = rearrange(w, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)  # [B,T,H,R,d]
            W = l2_norm(W)
            # if self.wavelet_baseline_use:
            #     qk = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2))
            #     rel = torch.einsum("blhd,dln->blhn", q, wavelet_decay_table)
            #     rel= rel.transpose(1, 2)
            #     wavelet_bias = (qk + rel) / torch.full(
            #         [], self.head_dim ** 0.5, dtype=q.dtype, device=q.device
            #     )
            #     mask_value = torch.finfo(wavelet_bias.dtype).min
            #     # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            #     # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            #     mask_value = torch.full([], mask_value, dtype=wavelet_bias.dtype, device=wavelet_bias.device)
            #     wavelet_bias = torch.where(build_causal_mask(rel.size(-1),rel.size(-1), device='cuda'), wavelet_bias.to(wavelet_bias.dtype), mask_value)
            #     wavelet_bias = nn.functional.softmax(wavelet_bias, dim=-1)
            #     wavelet_bias = self.attn_dropout(wavelet_bias)
            #     attn_output = torch.matmul(wavelet_bias, v.transpose(1, 2))
            w = rearrange(W, 'b t h r d -> b t (h r) d')                                   # [B,T,H*R,d]

            # === Wavelet(beta)（可选）===
            beta = torch.sigmoid(beta_logits) * 2.0
            if g is not None:
                g = rearrange(g, 'b t hq -> b t hq 1').repeat(1, 1, self.r, 1).view(g.shape[0], g.shape[1], -1)

            # 核心 op
            if not self.config.qk_rotation:
                if self.config.ablate_switch:
                    ablate = ablation_from_conflict_csv("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/top_conflict_heads.csv", topk=5, layer_whitelist=[0,6])
                else:
                    ablate = None

                # o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
                
                o = self.path_attention_with_wavelet_QH(
                    q=q, k=k, v=v,
                    w=w, beta=beta,
                    wavelet_dtt=wavelet_decay_table,
                    # wavelet_dtt=None,
                    use_wavelet_fused_H=False,   # 用 baseline PaTH 的 H（推荐先从这开始对齐）
                    d_chunk=8,
                    compute_dtype=torch.float32,
                    analyzer=analyzer,
                    # scale_wise_analyzer=scale_wise_analyzer,
                    layer_idx=self.layer_idx,
                    ablate=ablate,
                    rel_selection=self.config.rel_selection,
                    router1=router1 if router_active else None,
                    router2=router2 if router_active else None,
                    config=self.config,
                    global_step=global_step,
                    router_log_every=getattr(self.config, "router_log_every", 500),
                )                
            else:
                rot_q, rot_k = self.rotary_emb.rotate_queries_or_keys(q.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3), self.rotary_emb.rotate_queries_or_keys(k.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)
                o, _ = parallel_path_attn(q=rot_q, k=rot_k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
            # o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
            if self.layer_idx == 11 and analyzer:
                analyzer['layer_attention_analyzer'].save(
                    out_dir=f"analysis/{Path(self.config.model_name_or_path).name}_{Path(self.config.model_name_or_path).parent.name}",
                    tag="path_wavelet_QH",
                    make_plots=True,
                )         
                analyzer['scale_wise_analyzer'].save_npz()
                npz_path, fig_paths = analyzer['router_analyzer'].finalize_and_plot(prefix="router")
                analyzer['token_scale_dumper'].close()
                os._exit(0)
            # pdb.set_trace()
            if (self.layer_idx < self.config.distill_in_which_layers) and self.training:
                if self.config.temp_loss_coe != 0:
                    i_idx, j_idx, delta = sample_index_pairs(self.config.block_size, self.config.sample_num)
                    batch_idx = torch.randint(
                        low=0,
                        high=q.size(0),
                        size=(self.config.sample_num,),
                        device=q.device,
                    )
                with torch.no_grad():
                    if self.config.distill_teacher == 'rotary':
                        rot_q, rot_k = self.rotary_emb.rotate_queries_or_keys(q.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3), self.rotary_emb.rotate_queries_or_keys(k.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)
                        if self.config.temp_loss_coe != 0:
                            temp_teacher_scores = rot_q[batch_idx, j_idx, ...] * rot_k[batch_idx, i_idx, ...]
                        spectral_teacher_scores = rot_q[:, -1:, ...] * rot_k
                    elif self.config.distill_teacher == 'wavelet':
                        spectral_teacher_scores = compute_wavelet_scores_batched(q[:, -1, ...], k, wavelet_decay_table[:, -1, :])
                        if self.config.temp_loss_coe != 0:
                            temp_teacher_scores = compute_pair_wavelet_scores_batched(
                                q[batch_idx, j_idx, ...],
                                k[batch_idx, i_idx, ...],
                                wavelet_decay_table[batch_idx, i_idx, :],
                            )
                    elif self.config.distill_teacher == 'shrink':
                        with torch.no_grad():
                            spectral_teacher_scores = (q[:, -1:, ...] * k)
                            norm = spectral_teacher_scores.norm(dim=1, keepdim=True) + 1e-12
                            spectral_teacher_scores = spectral_teacher_scores / norm            
                    elif self.config.distill_teacher == 'shrink_w_shuffle':
                        with torch.no_grad():
                            spectral_teacher_scores = (q[:, -1:, ...] * k)
                            norm = spectral_teacher_scores.norm(dim=1, keepdim=True) + 1e-12
                            spectral_teacher_scores = spectral_teacher_scores / norm
                            spectral_teacher_scores = make_randomized_teacher_T(spectral_teacher_scores)
                    elif self.config.distill_teacher == 'mean_wavelet_pe':
                        spectral_teacher_scores = self.teacher_mean_spec[None, :, :, :]

                    else:
                        raise ValueError(f"Unknown distill_teacher: {self.config.distill_teacher}")
                if self.config.temp_loss_coe != 0:
                    temp_path_attn_scores = compute_path_score_multi(
                                                                        q, k, w, beta,
                                                                        i_idx=i_idx,
                                                                        j_idx=j_idx,
                                                                        batch_idx=batch_idx,
                                                                    )            
                    temp_loss = self.config.temp_loss_coe * F.mse_loss(temp_path_attn_scores, temp_teacher_scores)      
                else:
                    temp_loss = torch.tensor(0.0, device=q.device, dtype=q.dtype)
                path_attn_scores = path_attn_last_query_elementwise(q[:, -1:, ...], k, w, beta)
                K = q.size(1) // 2 + 1
                if self.config.weight_alpha > 0.0:
                    w = make_highfreq_weight(K, alpha=self.config.weight_alpha, device=q.device)
                    w = w[None, :, None, None]  # [1, K, 1, 1]
                else:
                    w = 1.0
                if self.config.distill_teacher == "mean_wavelet_pe":
                    if self.config.wavelet_pe_softmax_use:
                        path_attn_scores = F.softmax(path_attn_scores, dim=1)

                    # A_s: power spectrum, A_s_log: log spectrum
                    A_s, A_s_log = spectrum_over_T_multi(path_attn_scores.unsqueeze(1))
                    A_s_log_T = A_s_log.squeeze(1).transpose(1, 2)  # 和 teacher 对齐

                    eps = 1e-8
                    # teacher 从文件读的是 power mean_spectrum
                    # -> 在这里转成 log 频谱
                    A_t = spectral_teacher_scores  # [1, H, K, D]
                    A_t_log = torch.log(A_t + eps)

                    spectral_loss = self.config.spectral_loss_coe * F.mse_loss(
                        A_s_log_T, A_t_log
                    )
                else:
                    if self.config.loss_type == 'mse':
                        spectral_loss = self.config.spectral_loss_coe * spectral_distill_over_L_mse(path_attn_scores.unsqueeze(1), spectral_teacher_scores.unsqueeze(1) if spectral_teacher_scores.dim() == 4 else spectral_teacher_scores, w=w,lambda_kl=0.0, lambda_mse=1.0)
                    elif self.config.loss_type == 'cos':
                        spectral_loss = self.config.spectral_loss_coe * spectral_distill_over_L_cos(path_attn_scores.unsqueeze(1), spectral_teacher_scores.unsqueeze(1) if spectral_teacher_scores.dim() == 4 else spectral_teacher_scores, w=w)
                    elif self.config.loss_type == 'band_seperate_cos':
                        spectral_loss = self.config.spectral_loss_coe * spectral_distill_over_L_cos_3bands(path_attn_scores.unsqueeze(1), spectral_teacher_scores.unsqueeze(1) if spectral_teacher_scores.dim() == 4 else spectral_teacher_scores)
                dis_loss = temp_loss + spectral_loss
            else:
                dis_loss = q.sum() * 0.0
                # dis_loss = torch.tensor(0.0, device=q.device, dtype=q.dtype)
            o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
            o = self.o_proj(o)
            # if analyzer is not None:
            return o, None, past_key_values, dis_loss, record_router1, record_router2
            # else:
                # return o, None, past_key_values, dis_loss, None, None

        # ========= 其它路径（mask!=None）：最小实现 =========
        if self.use_w_shortconv:
            w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        q = rearrange(q, 'b t (h d) -> b t hq d', d=self.head_dim)
        k = rearrange(k, 'b t (h d) -> b t h d',  d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b t h d',  d=self.head_dim)
        W = rearrange(w, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)
        W = l2_norm(W)
        q = q.repeat_interleave(self.r, dim=2)
        k = k.repeat_interleave(self.r, dim=2)
        v = v.repeat_interleave(self.r, dim=2)
        w = rearrange(W, 'b t h r d -> b t (h r) d')

        # 先得到 beta（最小实现里不叠 wavelet；如需，可复用上面的代码段）
        beta = torch.sigmoid(beta_logits) * 2.0

        if g is not None:
            g = rearrange(g, 'b t hq -> b t hq 1').repeat(1, 1, self.r, 1).view(g.shape[0], g.shape[1], -1)

        o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
        o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
        o = self.o_proj(o)
        return o, None, past_key_values
# class PaTHAttention(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int = 2048,
#         num_heads: int = 32,
#         num_kv_heads: Optional[int] = None,
#         use_forget_gate: bool = False,
#         use_qk_norm: bool = False,
#         layer_idx: int = None,
#         use_low_rank_w: bool = True,
#         use_w_shortconv: bool = True,
#         conv_size: int = 3,
#         conv_bias: bool = False,
#         num_harmonics=1,
#     ):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         if num_kv_heads is None:
#             self.num_kv_heads = self.num_heads
#         else:
#             self.num_kv_heads = num_kv_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.kv_dim = self.num_kv_heads * self.head_dim

#         self.layer_idx = layer_idx

#         self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
#         self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
#         self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

#         # We use low-rank parameterization for the w_proj to reduce parameters in MHA settings.
#         if use_low_rank_w:
#             self.w_proj = nn.Sequential(
#                 nn.Linear(self.hidden_size, 32, bias=False),
#                 nn.Linear(32, self.kv_dim, bias=False)
#             )
#         # In MQA/GQA settings, key/value heads are shared, so we use a standard linear projection
#         # which doesn't introduce too many parameters
#         else:
#             self.w_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

#         # TODO: per head norm?
#         if use_qk_norm:
#             self.maybe_q_norm = RMSNorm(self.hidden_size)
#             self.maybe_k_norm = RMSNorm(self.kv_dim)
#         else:
#             self.maybe_q_norm = nn.Identity()
#             self.maybe_k_norm = nn.Identity()

#         if use_w_shortconv:
#             self.w_conv1d = ShortConvolution(hidden_size=self.kv_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
#         self.use_w_shortconv = use_w_shortconv
#         self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=True)
#         self.use_forget_gate = use_forget_gate
#         if use_forget_gate:
#             self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
#         self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if use_cache:
#             assert past_key_values is not None, "past_key_values must be provided when use_cache is True"
#         if attention_mask is not None:
#             assert len(attention_mask.shape) == 2, (
#                 "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
#                 "for padding purposes (0 indicating padding). "
#                 "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
#             )
#         batch_size, q_len, _ = hidden_states.size()
#         q = self.q_proj(hidden_states)
#         k = self.k_proj(hidden_states)
#         v = self.v_proj(hidden_states)
#         w = self.w_proj(hidden_states)
#         beta = self.bt_proj(hidden_states).sigmoid() * 2  # allowing negative eigenvalues
#         g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None
#         q, k = self.maybe_q_norm(q), self.maybe_k_norm(k)
#         cu_seqlens = kwargs.get('cu_seqlens', None)
#         assert not (cu_seqlens is not None and attention_mask is not None), (
#             "cu_seqlens should not be provided when attention_mask is not None"
#         )
#         # Training
#         if attention_mask is None:
#             assert use_cache is False, "use_cache should be False in training"
#             if self.use_w_shortconv:
#                 w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
#             q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
#             k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
#             v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
#             w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
#             w = l2_norm(w)
#             o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)

#         # Prefilling or decoding
#         else:
#             assert self.training is False, "attention mask is not supported in training. Please use variable length input."
#             try:
#                 last_state = past_key_values[self.layer_idx]
#             except KeyError:
#                 last_state = None
#             # Decoding
#             if last_state is not None:
#                 if g is not None:
#                     past_k, past_v, past_g = last_state['attn_state']
#                 else:
#                     past_k, past_v = last_state['attn_state']
#                 w_conv_state = last_state['conv_state']
#                 past_k = rearrange(past_k, '... (h d) -> ... h d', d=self.head_dim)
#                 if self.use_w_shortconv:
#                     w, w_conv_state = self.w_conv1d(w, cache=w_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
#                 w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
#                 w = l2_norm(w)

#                 @torch.compile
#                 def rank_one_update(k, w, beta):
#                     original_dtype = k.dtype
#                     k = k.float()
#                     w = w.float()
#                     beta = beta.float()
#                     k = k - beta[..., None].float() * (k * w).sum(-1, keepdim=True) * w
#                     return k.to(original_dtype)

#                 past_k = rank_one_update(past_k, w, beta)
#                 past_k = rearrange(past_k, '... h d -> ... (h d)')
#                 k = torch.cat([past_k, k], dim=1)
#                 v = torch.cat([past_v, v], dim=1)
#                 g = torch.cat([past_g, g], dim=1) if g is not None else None
#                 past_key_values[self.layer_idx]['attn_state'] = (k, v, g) if g is not None else (k, v)
#                 past_key_values.update(
#                     conv_state=w_conv_state,
#                     layer_idx=self.layer_idx,
#                     offset=q_len
#                 )
#                 if g is not None:
#                     q, (k, v, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
#                         q, (k, v, g), attention_mask, q_len, keepdim=True)
#                     max_seqlen_q, max_seqlen_k = max_seq_lens
#                 else:
#                     q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
#                         q, (k, v), attention_mask, q_len, keepdim=True)
#                     max_seqlen_q, max_seqlen_k = max_seq_lens
#                 _, cu_seqlens = cu_seqlens
#                 q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
#                 k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
#                 v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
#                 assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
#                 o = attn_decoding_one_step(q, k, v, g, cu_seqlens=cu_seqlens, do_gate_scale=True)  # reduced to fox's decoding
#             # Prefilling
#             else:
#                 v_cache = v.clone()
#                 g_cache = g.clone() if g is not None else None
#                 if g is None:
#                     q, (k, v, w, beta), indices_q, cu_seqlens, max_seq_lens = unpad_input(
#                         q, (k, v, w, beta), attention_mask, q_len, keepdim=True)
#                 else:
#                     q, (k, v, w, beta, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
#                         q, (k, v, w, beta, g), attention_mask, q_len, keepdim=True)
#                 max_seqlen_q, max_seqlen_k = max_seq_lens
#                 assert max_seqlen_q == max_seqlen_k, "max_seqlen_q should be equal to max_seqlen_k in prefilling"
#                 _, cu_seqlens = cu_seqlens
#                 if self.use_w_shortconv:
#                     w, w_conv_state = self.w_conv1d(w, cache=None, output_final_state=use_cache, cu_seqlens=cu_seqlens)
#                 else:
#                     w_conv_state = None
#                 q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
#                 k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
#                 v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
#                 w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
#                 w = l2_norm(w)
#                 o, k_cache = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g,
#                                                 cu_seqlens=cu_seqlens, use_cache=use_cache)
#                 if use_cache:
#                     k_cache = pad_input(k_cache.squeeze(0), indices_q, batch_size, q_len)
#                     k_cache = rearrange(k_cache, '... h d -> ... (h d)')
#                     past_key_values.update(
#                         attn_state=(k_cache, v_cache, g_cache) if g_cache is not None else (k_cache, v_cache),
#                         conv_state=w_conv_state,
#                         layer_idx=self.layer_idx,
#                         offset=q_len
#                     )
#             o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
#         o = rearrange(o, '... h d -> ... (h d)')
#         o = self.o_proj(o)
#         return o, None, past_key_values
class PaTHAttentionWfreq(nn.Module):
    def __init__(self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        use_forget_gate: bool = False,
        use_qk_norm: bool = False,
        layer_idx: int = None,
        use_low_rank_w: bool = True,
        use_w_shortconv: bool = True,
        conv_size: int = 3,
        conv_bias: bool = False,
        # NEW ↓↓↓
        num_harmonics: int = 2,
        share_freq_across_heads: bool = True,
        single_A_B: bool = False,
        use_beta_modulation: bool = False,
        use_wavelet_beta: bool = False,
        wavelet_mode: str = "additive",   # "additive" | "softmix"
    ):
        super().__init__()
        self.use_wavelet_beta = False
        self.use_beta_modulation = use_beta_modulation 
        self.single_A_B = single_A_B
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.r = num_harmonics
        self.use_w_shortconv = use_w_shortconv
        self.share_freq_across_heads = share_freq_across_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        # We use low-rank parameterization for the w_proj to reduce parameters in MHA settings.
        # 
        out_w_branch = self.num_kv_heads * self.head_dim if self.single_A_B else self.num_kv_heads * self.r * self.head_dim
        if use_low_rank_w:
            self.wA_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w_branch, bias=False),
            )
            self.wB_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w_branch, bias=False),
            )            
        else:
            self.wA_proj = nn.Linear(self.hidden_size, out_w_branch, bias=False)
            self.wB_proj = nn.Linear(self.hidden_size, out_w_branch, bias=False)

        # TODO: per head norm?
        if use_qk_norm:
            self.maybe_q_norm = RMSNorm(self.hidden_size)
            self.maybe_k_norm = RMSNorm(self.kv_dim)
        else:
            self.maybe_q_norm = nn.Identity()
            self.maybe_k_norm = nn.Identity()

        if use_w_shortconv:
            # 卷积的通道数也改成 2R 倍
            self.wA_conv1d = ShortConvolution(
                hidden_size=out_w_branch, kernel_size=conv_size, bias=conv_bias, activation='silu'
            )
            self.wB_conv1d = ShortConvolution(
                hidden_size=out_w_branch, kernel_size=conv_size, bias=conv_bias, activation='silu'
            )          
        self.use_w_shortconv = use_w_shortconv
        # β: 每个频带一个 β 门（范围 0~2）
        self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.r, bias=True)

        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        self.o_proj = nn.Linear(self.hidden_size * self.r, self.hidden_size, bias=False)
        Hf = 1 if share_freq_across_heads else self.num_kv_heads
        # 频率/相位参数：按开关决定 Hf；初始化为“近 0 频率、相位 0”
        # self.omega_raw = nn.Parameter(torch.full((1,1,Hf,self.r), -8.0))  # ~0
        # self.omega_scale = 2*math.pi / (512 * 16.0)
        # self.phi = nn.Parameter(torch.zeros(1,1,Hf,self.r))\
        self.omega_raw = nn.Parameter(torch.full((1,1,Hf,self.r), -8.0))

            # “底数为2”的无上界指数
        self.register_buffer("omega_base", torch.tensor(2*math.pi / 512.0))  # 你的 T_ref，如 512
        self.log2e = math.log(2.0)

        # ----- 相位参数：周期内比例 -----

        H = self.num_kv_heads
        # 12个头：e=[0,0,2,2,4,4,6,6,8,8,10,10]
        exp_list   = [2 * (h // 2) for h in range(H)]         # [0,0,2,2,...,10,10]
        shift_list = [float(h % 2) for h in range(H)]         # [0,1,0,1,...,1]

        # 形状 [1,1,H,1] 再在 r 维复制
        exp_init   = torch.tensor(exp_list, dtype=torch.float32).view(1,1,H,1).repeat(1,1,1,self.r)
        shift_init = torch.tensor(shift_list, dtype=torch.float32).view(1,1,H,1).repeat(1,1,1,self.r)

        # ★ e 是可学习参数；scale = 2**e 在 forward 里计算
        self.ricker_scale_exp = nn.Parameter(exp_init)        # [1,1,H,r]  ← 学这个“指数”
        self.ricker_shift     = nn.Parameter(shift_init)      # [1,1,H,r]  ← 直接学 shift（按 token）
        self.mix_logit = nn.Parameter(torch.tensor(0.0)) if wavelet_mode == "softmix" else None
    def get_omega(self):
        raw = self.omega_raw.clamp(min=-20, max=20)
        return self.omega_base * torch.exp(raw * self.log2e)
    def get_phi(self):
        # φ = 2π * τ, τ∈[0,1)
        tau = torch.sigmoid(self.phi_raw)
        return 2*math.pi * tau
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert self.hidden_size == self.num_heads * self.head_dim
        if self.single_A_B:
            assert self.wA_proj[-1].out_features == self.num_kv_heads * self.head_dim  # low-rank情形
        else:
            assert self.wA_proj[-1].out_features == self.num_kv_heads * self.r * self.head_dim        
        if use_cache:
            assert past_key_values is not None, "past_key_values must be provided when use_cache is True"
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert not (cu_seqlens is not None and attention_mask is not None), (
            "cu_seqlens should not be provided when attention_mask is not None"
        )
        # Training
        B, T, _ = hidden_states.size()
        q_flat = self.q_proj(hidden_states)                  # [B,T,hidden_size]
        k_flat = self.k_proj(hidden_states)                  # [B,T,kv_dim]
        v_flat = self.v_proj(hidden_states)                  # [B,T,kv_dim]
        q_flat, k_flat = self.maybe_q_norm(q_flat), self.maybe_k_norm(k_flat)

        # FoX/遗忘门（如果启用）：此处只算一次
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None

        # === 位置与谐振 ===


        # === 生成 wA/wB（可选短卷积）===
        wA = self.wA_proj(hidden_states)
        wB = self.wB_proj(hidden_states)
        if self.use_w_shortconv:
            wA, _ = self.wA_conv1d(wA, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
            wB, _ = self.wB_conv1d(wB, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

        # === ① 计算 w（方向）===
        if not self.use_beta_modulation:
            pos = torch.arange(T, device=hidden_states.device).view(1, T, 1, 1)  # 整型 arange
            omega = self.get_omega()                 # [1,1,Hf,r]
            phi   = self.get_phi()                   # [1,1,Hf,r]
            theta = omega * pos + phi                # [1,T,Hf,r]
            c = torch.cos(theta).to(hidden_states.dtype)  # [1,T,Hf,r]            
            if self.single_A_B:
                A = rearrange(wA, 'b t (h d) -> b t h d', h=self.num_kv_heads, d=self.head_dim).unsqueeze(3)
                B_ = rearrange(wB, 'b t (h d) -> b t h d', h=self.num_kv_heads, d=self.head_dim).unsqueeze(3)
                A = A.repeat_interleave(self.r, dim=-2)
                B_ = B_.repeat_interleave(self.r, dim=-2)
            else:
                A  = rearrange(wA, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)
                B_ = rearrange(wB, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)

# 先在最后一维显式加一维度，再 expand 到目标形状
            c_full = c[..., None].expand(B, T, self.num_kv_heads if c.size(2)==1 else c.size(2), self.r, 1)  # [B,T,H,R,1]
            s_full = torch.sin(theta).to(hidden_states.dtype)[..., None].expand_as(c_full)                   # [B,T,H,R,1]

            W = A * c_full + B_ * s_full                              # [B,T,H,R,d]
            W = l2_norm(W)
            w = rearrange(W, 'b t h r d -> b t (h r) d')              # [B,T,H*R,d]
        else:
            pos = torch.arange(T, device=hidden_states.device).view(1, T, 1, 1)  # 整型 arange
            omega = self.get_omega()                 # [1,1,Hf,r]
            phi   = self.get_phi()                   # [1,1,Hf,r]
            theta = omega * pos + phi                # [1,T,Hf,r]
            c = torch.cos(theta).to(hidden_states.dtype)  # [1,T,Hf,r]
            if self.single_A_B:
                Wbase = rearrange(wA, 'b t (h d) -> b t h d', h=self.num_kv_heads, d=self.head_dim)  # [B,T,H,d]
            else:
                Wtmp  = rearrange(wA, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)
                Wbase = Wtmp.mean(dim=3)                                                                    # [B,T,H,d]
            Wbase = l2_norm(Wbase)
            w = repeat(Wbase, 'b t h d -> b t (h r) d', r=self.r)

        # === ② 重排 q/k/v 到多头并复制到 R ===
        q = rearrange(q_flat, 'b t (h d) -> b t h d', d=self.head_dim)
        k = rearrange(k_flat, 'b t (h d) -> b t h d', d=self.head_dim)
        v = rearrange(v_flat, 'b t (h d) -> b t h d', d=self.head_dim)
        q = q.repeat_interleave(self.r, dim=2)
        k = k.repeat_interleave(self.r, dim=2)
        v = v.repeat_interleave(self.r, dim=2)

        # === ③ β（忘记门）===
        beta_logits = self.bt_proj(hidden_states)                      # [B,T,H*R]
        if self.use_beta_modulation:
            Hf = c.size(2)                                            # 1 or H
            amp = self.beta_amp                                       # [1,1,Hf,r]
            freq = (amp * c).to(beta_logits.dtype)                    # [1,T,Hf,r]
            if Hf == 1:
                freq = freq.expand(B, T, self.num_kv_heads, self.r)   # [B,T,H,r]
            else:
                freq = freq.expand(B, T, Hf, self.r)
            freq = rearrange(freq, 'b t h r -> b t (h r)')            # [B,T,H*R]
            beta_logits = beta_logits + freq
        if getattr(self, "use_wavelet_beta", False):
            # 末位对齐：pos ∈ [-T+1, ..., 0]，0 处峰值
            pos_end = torch.arange(-T+1, 1, device=hidden_states.device).view(1, T, 1, 1).to(hidden_states.dtype)  # [1,T,1,1]

            # σ（token 级，带下限）
            # sigma = F.softplus(self.ricker_log_s) + self.sigma_min              # [1,1,H,r]

            # === 指数项可学：scale = 2**e ===
            e = self.ricker_scale_exp.to(hidden_states.dtype)                   # [1,1,H,r]
            # 可选护栏：e = e.clamp(-2, 12)
            scale = torch.exp2(e)                                               # [1,1,H,r]
            shift = self.ricker_shift.to(hidden_states.dtype)                   # [1,1,H,r]

            # 扩展到 [B,T,H,r]
            scale = scale.expand(1, T, self.num_kv_heads, self.r).expand(B, T, self.num_kv_heads, self.r)
            shift = shift.expand(1, T, self.num_kv_heads, self.r).expand_as(scale)
            # sigma_full = sigma.expand(1, T, self.num_kv_heads, self.r).expand_as(scale)

            # 仿射时间轴：t_affine = scale * (pos_end - shift)
            t_affine = scale * (pos_end - shift)                                # [B,T,H,r]

            # Ricker
            # tau = t_affine / (sigma_full + 1e-6)
            psi = (1.0 - t_affine**2) * torch.exp(-0.5 * t_affine**2)                    # [B,T,H,r]
            psi = psi - psi.mean(dim=1, keepdim=True)

            wave = (self.ricker_amp.to(beta_logits.dtype).expand_as(psi) * psi).to(beta_logits.dtype)  # [B,T,H,r]
            wave = rearrange(wave, 'b t h r -> b t (h r)')                      # [B,T,H*R]
            if self.wavelet_mode == "additive":
                beta_logits = beta_logits + wave
                beta = torch.sigmoid(beta_logits) * 2.0
            elif self.wavelet_mode == "softmix":
                beta_base = torch.sigmoid(beta_logits) * 2.0
                beta_gate = torch.sigmoid(beta_logits + wave) * 2.0
                lam = torch.sigmoid(self.mix_logit)
                beta = (1 - lam) * beta_base + lam * beta_gate
            else:
                raise ValueError(f"Unknown wavelet_mode: {self.wavelet_mode}")
        else:
            beta = torch.sigmoid(beta_logits) * 2.0

        # === ④ FoX 门形状扩展（如启用，不要重算 g）===
        if g is not None:
            g = rearrange(g, 'b t h -> b t h 1').repeat(1,1,self.r,1).squeeze(-1)  # [B,T,H*R]

        # === ⑤ Path Attention ===
        o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g,
                                cu_seqlens=cu_seqlens, use_cache=False)
        o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
        o = self.o_proj(o)
        return o, None, past_key_values
