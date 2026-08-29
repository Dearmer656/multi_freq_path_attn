# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
from collections import Counter, deque
import csv
import hashlib
import json

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
from fla.ops.entmax import masked_entmax_bisect
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
from transformers import AutoTokenizer
from transformers.activations import NewGELUActivation

import pdb
import os
import torch
from tqdm import tqdm

SCALE_MULTIPLIER_DICT = {
    'wavelet': 1.0,
    'morlet': 3.0249,
    'gaussian': 0.5316,
    'linear': 0.7228,
    # PAT-164 basis ablation never calibrated a half-life multiplier for sine
    # (periodic, no natural width to match) -- left unscaled at 1.0.
    'sine': 1.0,
    # Deliberately left at 1.0 (same as 'wavelet'), NOT re-derived like morlet's
    # 3.0249 -- ricker_cos reuses the Ricker envelope verbatim and only adds a
    # cos carrier on top, so it must share Ricker's exact effective scale to
    # isolate "does the oscillation help" from "does the envelope width differ".
    'ricker_cos': 1.0,
    # PAT-225: morlet's 3.0249 multiplier is calibrated by matching the FULL
    # function's (envelope * cos) zero-crossing to Ricker's zero at u=1 -- but
    # morlet's envelope alone (exp(-0.5u^2)) is identical to 'gaussian', whose
    # own multiplier is 0.5316. Using morlet's 3.0249 makes its envelope ~5.7x
    # WIDER than a same-named Ricker/gaussian config (verified: rho=256 Ricker
    # -> effective 256.0, rho=256 Morlet -> effective 774.4). morlet_gaussamp
    # reuses morlet's exact basis fn (envelope * cos) but with gaussian's
    # multiplier, so the envelope half-life matches Ricker's at the same exp.
    'morlet_gaussamp': 0.5316,
}


def _tensor_debug_summary_json(name: str, x: Optional[torch.Tensor]) -> Optional[dict]:
    if x is None or (not torch.is_tensor(x)):
        return None
    xt = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
    finite = torch.isfinite(xt)
    out = {
        "name": str(name),
        "shape": list(xt.shape),
        "numel": int(xt.numel()),
        "finite_ratio": float(finite.float().mean().item()) if xt.numel() > 0 else float("nan"),
        "abs_sum": float(torch.nan_to_num(xt.abs(), nan=0.0, posinf=0.0, neginf=0.0).sum().item()),
    }
    if int(xt.numel()) > 0 and bool(finite.any()):
        xf = xt[finite]
        out.update(
            {
                "mean": float(xf.mean().item()),
                "std": float(xf.std(unbiased=False).item()),
                "min": float(xf.min().item()),
                "max": float(xf.max().item()),
            }
        )
    try:
        out["md5"] = hashlib.md5(xt.numpy().tobytes()).hexdigest()
    except Exception:
        pass
    return out


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
# NOTE: this module previously defined sample_index_pairs() twice (a "mix"-default
# earlier copy was silently shadowed by this one, which also adds i_bias_power).
# The duplicate was removed 2026-07-18; this is now the single source of truth.
# Its only current call site (temp_loss_coe distillation sampling) is inert in
# every PAT-225/226/227 config (temp_loss_coe=0, distill_in_which_layers=0).
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
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.numel() == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = torch.quantile(y, q).item()
    return out

@torch.no_grad()
def _monitor_tensor_stats_bthd(
    x: Optional[torch.Tensor],
    *,
    max_tokens: int = 64,
    max_heads: int = 4,
):
    if x is None:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "abs_p99": float("nan"),
            "norm_p50": float("nan"),
            "norm_p90": float("nan"),
            "norm_p99": float("nan"),
        }
    y = x.detach().float()
    if y.dim() == 4:
        _, T, H, D = y.shape
        t_take = max(1, min(int(max_tokens), int(T)))
        h_take = max(1, min(int(max_heads), int(H)))
        if t_take < T:
            t_idx = torch.linspace(0, T - 1, steps=t_take, device=y.device).long()
            y = y.index_select(1, t_idx)
        if h_take < H:
            h_idx = torch.linspace(0, H - 1, steps=h_take, device=y.device).long()
            y = y.index_select(2, h_idx)
        vec = y.reshape(-1, D)
    else:
        flat_tmp = y.reshape(-1)
        vec = flat_tmp.unsqueeze(-1)
    flat = y.reshape(-1)
    if flat.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "abs_p99": float("nan"),
            "norm_p50": float("nan"),
            "norm_p90": float("nan"),
            "norm_p99": float("nan"),
        }
    abs_q = _quantiles_flat(flat.abs(), qs=(0.99,))
    norm = torch.linalg.vector_norm(vec, ord=2, dim=-1)
    norm_q = _quantiles_flat(norm, qs=(0.5, 0.9, 0.99))
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "abs_p99": float(abs_q["p99"]),
        "norm_p50": float(norm_q["p50"]),
        "norm_p90": float(norm_q["p90"]),
        "norm_p99": float(norm_q["p99"]),
    }

@torch.no_grad()
def _monitor_scalar_stats(x: Optional[torch.Tensor]):
    if x is None:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    q = _quantiles_flat(y, qs=(0.5, 0.9, 0.99))
    return {
        "mean": float(y.mean().item()),
        "p50": float(q["p50"]),
        "p90": float(q["p90"]),
        "p99": float(q["p99"]),
    }

@torch.no_grad()
def _monitor_flat_stats(x: Optional[torch.Tensor]):
    if x is None:
        return {"mean": float("nan"), "std": float("nan"), "abs_p99": float("nan")}
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "abs_p99": float("nan")}
    q = _quantiles_flat(y.abs(), qs=(0.99,))
    return {
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "abs_p99": float(q["p99"]),
    }

@torch.no_grad()
def _monitor_attn_prob_stats(
    attn_probs: Optional[torch.Tensor],
    *,
    max_queries: int = 64,
    max_heads: int = 4,
):
    if attn_probs is None:
        return {
            "entropy_mean": float("nan"),
            "entropy_p50": float("nan"),
            "entropy_p90": float("nan"),
            "entropy_p99": float("nan"),
            "top1_mean": float("nan"),
            "top1_p50": float("nan"),
            "top1_p90": float("nan"),
            "top1_p99": float("nan"),
            "margin_mean": float("nan"),
            "margin_p50": float("nan"),
            "margin_p90": float("nan"),
            "margin_p99": float("nan"),
        }
    p = attn_probs.detach().float()
    if p.dim() != 4:
        return {
            "entropy_mean": float("nan"),
            "entropy_p50": float("nan"),
            "entropy_p90": float("nan"),
            "entropy_p99": float("nan"),
            "top1_mean": float("nan"),
            "top1_p50": float("nan"),
            "top1_p90": float("nan"),
            "top1_p99": float("nan"),
            "margin_mean": float("nan"),
            "margin_p50": float("nan"),
            "margin_p90": float("nan"),
            "margin_p99": float("nan"),
        }

    _, H, T, _ = p.shape
    h_take = max(1, min(int(max_heads), int(H)))
    q_take = max(1, min(int(max_queries), int(T)))
    if h_take < H:
        h_idx = torch.linspace(0, H - 1, steps=h_take, device=p.device).long()
        p = p.index_select(1, h_idx)
    if q_take < T:
        q_idx = torch.linspace(0, T - 1, steps=q_take, device=p.device).long()
        p = p.index_select(2, q_idx)

    dist = p.reshape(-1, p.shape[-1]).clamp_min(1e-12)
    ent = -(dist * dist.log()).sum(dim=-1)
    top2 = torch.topk(dist, k=min(2, dist.shape[-1]), dim=-1).values
    top1 = top2[..., 0]
    if top2.shape[-1] > 1:
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = torch.zeros_like(top1)

    q_ent = _quantiles_flat(ent, qs=(0.5, 0.9, 0.99))
    q_top1 = _quantiles_flat(top1, qs=(0.5, 0.9, 0.99))
    q_margin = _quantiles_flat(margin, qs=(0.5, 0.9, 0.99))
    return {
        "entropy_mean": float(ent.mean().item()),
        "entropy_p50": float(q_ent["p50"]),
        "entropy_p90": float(q_ent["p90"]),
        "entropy_p99": float(q_ent["p99"]),
        "top1_mean": float(top1.mean().item()),
        "top1_p50": float(q_top1["p50"]),
        "top1_p90": float(q_top1["p90"]),
        "top1_p99": float(q_top1["p99"]),
        "margin_mean": float(margin.mean().item()),
        "margin_p50": float(q_margin["p50"]),
        "margin_p90": float(q_margin["p90"]),
        "margin_p99": float(q_margin["p99"]),
    }


class WaveletCondFiLMv2StatsMeter:
    def __init__(self, window: int = 200):
        self.window = max(1, int(window))
        self.sat_s = deque(maxlen=self.window)
        self.sat_t = deque(maxlen=self.window)
        self.grad_s = deque(maxlen=self.window)
        self.grad_t = deque(maxlen=self.window)
        self.broken = deque(maxlen=self.window)

    @staticmethod
    def _median_finite(vals):
        finite = [float(v) for v in vals if math.isfinite(float(v))]
        if len(finite) == 0:
            return float("nan")
        finite.sort()
        n = len(finite)
        m = n // 2
        if (n % 2) == 1:
            return float(finite[m])
        return float(0.5 * (finite[m - 1] + finite[m]))

    def update(self, *, sat_s: float, sat_t: float, grad_s: float, grad_t: float, broken: bool):
        self.sat_s.append(float(sat_s))
        self.sat_t.append(float(sat_t))
        self.grad_s.append(float(grad_s))
        self.grad_t.append(float(grad_t))
        self.broken.append(bool(broken))

    def detect(self, *, sat_thresh: float = 0.7, grad_eps: float = 1e-6):
        if len(self.sat_s) == 0:
            return False, False, {}
        broken_window = any(bool(x) for x in self.broken)
        sat_s_mean = float(sum(self.sat_s) / max(1, len(self.sat_s)))
        sat_t_mean = float(sum(self.sat_t) / max(1, len(self.sat_t)))
        grad_s_med = self._median_finite(self.grad_s)
        grad_t_med = self._median_finite(self.grad_t)
        locked_window = (
            len(self.sat_s) >= self.window
            and sat_s_mean > float(sat_thresh)
            and sat_t_mean > float(sat_thresh)
            and math.isfinite(grad_s_med)
            and math.isfinite(grad_t_med)
            and grad_s_med < float(grad_eps)
            and grad_t_med < float(grad_eps)
        )
        return bool(broken_window), bool(locked_window), {
            "sat_s_mean": sat_s_mean,
            "sat_t_mean": sat_t_mean,
            "grad_s_med": grad_s_med,
            "grad_t_med": grad_t_med,
        }


class WaveletCondFiLM_v2(nn.Module):
    """
    Query-conditioned, bounded FiLM modulation for attention outputs.
    Default: per-token scalar scale/shift applied as out = out * scale + shift.
    """

    def __init__(
        self,
        d_model: int,
        d_wavelet: Optional[int] = None,
        hidden: int = 128,
        alpha: float = 0.1,
        beta: float = 0.1,
        clamp: float = 8.0,
        per_token_scalar: bool = True,
        print_every: int = 100,
        lock_window: int = 200,
        lock_sat_thresh: float = 0.7,
        lock_grad_eps: float = 1e-6,
        update_eps: float = 1e-8,
        grad_clip_value: float = 0.0,
        use_full_backward_hook: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_wavelet = None if d_wavelet is None else int(d_wavelet)
        self.hidden = max(1, int(hidden))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.clamp = float(clamp)
        self.per_token_scalar = bool(per_token_scalar)
        self.print_every = max(1, int(print_every))
        self.lock_sat_thresh = float(lock_sat_thresh)
        self.lock_grad_eps = float(lock_grad_eps)
        self.update_eps = float(update_eps)
        self.grad_clip_value = max(0.0, float(grad_clip_value))
        self.use_full_backward_hook = bool(use_full_backward_hook)

        out_dim = 1 if self.per_token_scalar else self.d_model
        in_dim = self.d_model + (self.d_wavelet if self.d_wavelet is not None else 0)

        self.ln_q = nn.LayerNorm(self.d_model)
        self.ln_w = nn.LayerNorm(self.d_wavelet) if self.d_wavelet is not None else None
        self.fc = nn.Linear(in_dim, self.hidden)
        self.linear_s = nn.Linear(self.hidden, out_dim)
        self.linear_t = nn.Linear(self.hidden, out_dim)

        nn.init.normal_(self.linear_s.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.linear_t.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_t.bias)

        self._last_forward = {}
        self._last_grads = {
            "s_raw_finite": None,
            "t_raw_finite": None,
            "scale_finite": None,
            "shift_finite": None,
            "s_raw_abs": float("nan"),
            "t_raw_abs": float("nan"),
            "scale_abs": float("nan"),
            "shift_abs": float("nan"),
            "lin_s_w_finite": None,
            "lin_t_w_finite": None,
            "lin_s_w_abs": float("nan"),
            "lin_t_w_abs": float("nan"),
            "module_bwd_finite": None,
        }
        self._prev_w_s = None
        self._prev_w_t = None
        self._prev_step = None
        self._last_update = {
            "s_mean": float("nan"),
            "s_p90": float("nan"),
            "t_mean": float("nan"),
            "t_p90": float("nan"),
        }
        self._local_step = 0
        self._last_log_step = None
        self._last_warn_step = None
        self._last_attn_stats = None
        self.stats_meter = WaveletCondFiLMv2StatsMeter(window=lock_window)

        self.linear_s.weight.register_hook(self._make_grad_hook("lin_s_w"))
        self.linear_t.weight.register_hook(self._make_grad_hook("lin_t_w"))
        if self.use_full_backward_hook:
            self.register_full_backward_hook(self._module_backward_hook)

    @staticmethod
    def _to_int(step):
        if step is None:
            return None
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                return None
            step = step.detach().item()
        try:
            return int(step)
        except Exception:
            return None

    def _resolve_step(self, step):
        step_i = self._to_int(step)
        if step_i is None:
            self._local_step += 1
            step_i = int(self._local_step)
        return int(step_i)

    def _module_backward_hook(self, module, grad_input, grad_output):
        finite = True
        try:
            for g in list(grad_input) + list(grad_output):
                if isinstance(g, torch.Tensor):
                    finite = finite and bool(torch.isfinite(g.detach().float()).all().item())
        except Exception:
            finite = False
        self._last_grads["module_bwd_finite"] = bool(finite)
        return None

    def _make_grad_hook(self, key: str):
        def _hook(grad: torch.Tensor):
            g = grad.detach().float()
            finite = bool(torch.isfinite(g).all().item())
            abs_mean = float(g.abs().mean().item()) if finite else float("nan")
            if key == "lin_s_w":
                self._last_grads["lin_s_w_finite"] = finite
                self._last_grads["lin_s_w_abs"] = abs_mean
            elif key == "lin_t_w":
                self._last_grads["lin_t_w_finite"] = finite
                self._last_grads["lin_t_w_abs"] = abs_mean
            if self.grad_clip_value > 0.0 and finite:
                g_clip = g.clamp(min=-self.grad_clip_value, max=self.grad_clip_value)
                return g_clip.to(dtype=grad.dtype)
            return grad

        return _hook

    def _make_act_grad_hook(self, key: str):
        def _hook(grad: torch.Tensor):
            g = grad.detach().float()
            finite = bool(torch.isfinite(g).all().item())
            abs_mean = float(g.abs().mean().item()) if finite else float("nan")
            self._last_grads[f"{key}_finite"] = finite
            self._last_grads[f"{key}_abs"] = abs_mean
            if self.grad_clip_value > 0.0 and finite:
                g_clip = g.clamp(min=-self.grad_clip_value, max=self.grad_clip_value)
                return g_clip.to(dtype=grad.dtype)
            return grad

        return _hook

    def should_log(self, step=None):
        step_i = self._resolve_step(step)
        return (step_i % int(self.print_every)) == 0

    def _safe_corr(self, x: torch.Tensor, y: torch.Tensor):
        if x.numel() <= 1 or y.numel() <= 1:
            return float("nan")
        xc = x - x.mean()
        yc = y - y.mean()
        den = (torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)).clamp_min(1e-12)
        return float((xc * yc).sum().div(den).item())

    def _update_param_delta(self, step_i: int):
        w_s = self.linear_s.weight.detach().float()
        w_t = self.linear_t.weight.detach().float()
        if self._prev_w_s is None or self._prev_w_t is None:
            self._prev_w_s = w_s.clone()
            self._prev_w_t = w_t.clone()
            self._prev_step = int(step_i)
            self._last_update = {
                "s_mean": float("nan"),
                "s_p90": float("nan"),
                "t_mean": float("nan"),
                "t_p90": float("nan"),
            }
            return
        if self._prev_step == int(step_i):
            return

        rs = (w_s - self._prev_w_s).abs() / (self._prev_w_s.abs() + float(self.update_eps))
        rt = (w_t - self._prev_w_t).abs() / (self._prev_w_t.abs() + float(self.update_eps))
        qs = _quantiles_flat(rs.reshape(-1), qs=(0.9,))
        qt = _quantiles_flat(rt.reshape(-1), qs=(0.9,))
        self._last_update = {
            "s_mean": float(rs.mean().item()),
            "s_p90": float(qs["p90"]),
            "t_mean": float(rt.mean().item()),
            "t_p90": float(qt["p90"]),
        }
        self._prev_w_s = w_s.clone()
        self._prev_w_t = w_t.clone()
        self._prev_step = int(step_i)

    def forward(self, q_in: torch.Tensor, attn_out: torch.Tensor, w_ctx: Optional[torch.Tensor] = None):
        assert q_in.dim() == 3, f"q_in must be [B,T,D], got {tuple(q_in.shape)}"
        assert attn_out.dim() >= 3, f"attn_out must be [B,T,...], got {tuple(attn_out.shape)}"
        assert torch.is_floating_point(q_in), f"q_in must be floating, got {q_in.dtype}"
        assert torch.is_floating_point(attn_out), f"attn_out must be floating, got {attn_out.dtype}"
        assert q_in.shape[0] == attn_out.shape[0] and q_in.shape[1] == attn_out.shape[1], (
            tuple(q_in.shape),
            tuple(attn_out.shape),
        )
        assert q_in.shape[-1] == self.d_model, (q_in.shape[-1], self.d_model)
        if w_ctx is not None:
            assert w_ctx.dim() == 3, f"w_ctx must be [B,T,Dw], got {tuple(w_ctx.shape)}"
            assert w_ctx.shape[:2] == q_in.shape[:2], (tuple(w_ctx.shape), tuple(q_in.shape))
            if self.d_wavelet is not None:
                assert w_ctx.shape[-1] == self.d_wavelet, (w_ctx.shape[-1], self.d_wavelet)

        out_dtype = attn_out.dtype
        device_type = attn_out.device.type if isinstance(attn_out, torch.Tensor) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            q32 = q_in.to(dtype=torch.float32)
            out32 = attn_out.to(dtype=torch.float32)
            u = self.ln_q(q32)
            if w_ctx is not None:
                w32 = w_ctx.to(dtype=torch.float32)
                v = self.ln_w(w32) if self.ln_w is not None else w32
                z = torch.cat([u, v], dim=-1)
            else:
                z = u

            h = F.gelu(self.fc(z))
            s_raw = self.linear_s(h)
            t_raw = self.linear_t(h)
            s_raw_clamped = s_raw.clamp(min=-self.clamp, max=self.clamp)
            t_raw_clamped = t_raw.clamp(min=-self.clamp, max=self.clamp)
            scale = 1.0 + float(self.alpha) * torch.tanh(s_raw_clamped)
            shift = float(self.beta) * torch.tanh(t_raw_clamped)

            if self.training and torch.is_grad_enabled():
                if s_raw.requires_grad:
                    s_raw.register_hook(self._make_act_grad_hook("s_raw"))
                if t_raw.requires_grad:
                    t_raw.register_hook(self._make_act_grad_hook("t_raw"))
                if scale.requires_grad:
                    scale.register_hook(self._make_act_grad_hook("scale"))
                if shift.requires_grad:
                    shift.register_hook(self._make_act_grad_hook("shift"))

            scale_b = scale
            shift_b = shift
            while scale_b.dim() < out32.dim():
                scale_b = scale_b.unsqueeze(-1)
                shift_b = shift_b.unsqueeze(-1)
            out_mod = out32 * scale_b + shift_b

            sat_thr = max(0.0, float(self.clamp) - 0.5)
            sat_s = float((s_raw_clamped.detach().abs() > sat_thr).float().mean().item())
            sat_t = float((t_raw_clamped.detach().abs() > sat_thr).float().mean().item())

            s_flat = s_raw_clamped.detach().reshape(-1).float()
            t_flat = t_raw_clamped.detach().reshape(-1).float()
            sc_flat = scale.detach().reshape(-1).float()
            sh_flat = shift.detach().reshape(-1).float()
            s_q = _quantiles_flat(s_flat, qs=(0.5, 0.9))
            t_q = _quantiles_flat(t_flat, qs=(0.5, 0.9))
            sc_q = _quantiles_flat(sc_flat, qs=(0.5, 0.9))
            sh_q = _quantiles_flat(sh_flat, qs=(0.5, 0.9))

            q_abs_tok = q32.detach().abs().mean(dim=-1).reshape(-1)
            s_abs_tok = s_raw_clamped.detach().abs().mean(dim=-1).reshape(-1)
            corr_q_s = self._safe_corr(q_abs_tok, s_abs_tok)

            self._last_forward = {
                "nf_s_raw": int((not torch.isfinite(s_raw).all().item())),
                "nf_t_raw": int((not torch.isfinite(t_raw).all().item())),
                "nf_scale": int((not torch.isfinite(scale).all().item())),
                "nf_shift": int((not torch.isfinite(shift).all().item())),
                "sat_s": sat_s,
                "sat_t": sat_t,
                "s_mean": float(s_flat.mean().item()),
                "s_p50": float(s_q["p50"]),
                "s_p90": float(s_q["p90"]),
                "s_max": float(s_flat.max().item()),
                "t_mean": float(t_flat.mean().item()),
                "t_p50": float(t_q["p50"]),
                "t_p90": float(t_q["p90"]),
                "t_max": float(t_flat.max().item()),
                "scale_mean": float(sc_flat.mean().item()),
                "scale_p50": float(sc_q["p50"]),
                "scale_p90": float(sc_q["p90"]),
                "shift_mean": float(sh_flat.mean().item()),
                "shift_p50": float(sh_q["p50"]),
                "shift_p90": float(sh_q["p90"]),
                "corr_q_s": float(corr_q_s),
            }

        return out_mod.to(dtype=out_dtype)

    def set_attn_stats(self, attn_stats: Optional[dict]):
        self._last_attn_stats = attn_stats

    def _nf_flag(self, v):
        if v is None:
            return -1
        return 0 if bool(v) else 1

    def log_if_needed(self, *, step=None, layer_idx: Optional[int] = None, logger_obj=None):
        step_i = self._resolve_step(step)
        if self._last_log_step == step_i:
            return
        if (step_i % int(self.print_every)) != 0:
            return
        self._last_log_step = int(step_i)
        if len(self._last_forward) == 0:
            return

        self._update_param_delta(step_i)

        grad_s = float(self._last_grads.get("lin_s_w_abs", float("nan")))
        grad_t = float(self._last_grads.get("lin_t_w_abs", float("nan")))
        broken_now = bool(
            self._last_forward.get("nf_s_raw", 0)
            or self._last_forward.get("nf_t_raw", 0)
            or self._last_forward.get("nf_scale", 0)
            or self._last_forward.get("nf_shift", 0)
            or (self._last_grads.get("scale_finite") is False)
            or (self._last_grads.get("shift_finite") is False)
        )

        self.stats_meter.update(
            sat_s=float(self._last_forward["sat_s"]),
            sat_t=float(self._last_forward["sat_t"]),
            grad_s=grad_s,
            grad_t=grad_t,
            broken=broken_now,
        )
        broken_win, locked_win, lock_payload = self.stats_meter.detect(
            sat_thresh=float(self.lock_sat_thresh),
            grad_eps=float(self.lock_grad_eps),
        )

        lid = -1 if layer_idx is None else int(layer_idx)
        msg = (
            f"[wavelet condfilm_v2 stats] layer={lid} step={int(step_i)} "
            f"nf_fwd[s_raw,t_raw,scale,shift]={self._last_forward['nf_s_raw']},"
            f"{self._last_forward['nf_t_raw']},{self._last_forward['nf_scale']},{self._last_forward['nf_shift']} "
            f"nf_grad[s_raw,t_raw,scale,shift,lin_s,lin_t]={self._nf_flag(self._last_grads.get('s_raw_finite'))},"
            f"{self._nf_flag(self._last_grads.get('t_raw_finite'))},"
            f"{self._nf_flag(self._last_grads.get('scale_finite'))},"
            f"{self._nf_flag(self._last_grads.get('shift_finite'))},"
            f"{self._nf_flag(self._last_grads.get('lin_s_w_finite'))},"
            f"{self._nf_flag(self._last_grads.get('lin_t_w_finite'))} | "
            f"sat_s={self._last_forward['sat_s']:.6e} sat_t={self._last_forward['sat_t']:.6e} | "
            f"s_raw mean={self._last_forward['s_mean']:.6e} p50={self._last_forward['s_p50']:.6e} "
            f"p90={self._last_forward['s_p90']:.6e} max={self._last_forward['s_max']:.6e} | "
            f"t_raw mean={self._last_forward['t_mean']:.6e} p50={self._last_forward['t_p50']:.6e} "
            f"p90={self._last_forward['t_p90']:.6e} max={self._last_forward['t_max']:.6e} | "
            f"scale mean={self._last_forward['scale_mean']:.6e} p50={self._last_forward['scale_p50']:.6e} "
            f"p90={self._last_forward['scale_p90']:.6e} | "
            f"shift mean={self._last_forward['shift_mean']:.6e} p50={self._last_forward['shift_p50']:.6e} "
            f"p90={self._last_forward['shift_p90']:.6e} | "
            f"upd_ratio_s mean={self._last_update['s_mean']:.6e} p90={self._last_update['s_p90']:.6e} "
            f"upd_ratio_t mean={self._last_update['t_mean']:.6e} p90={self._last_update['t_p90']:.6e} | "
            f"grad_abs lin_s={grad_s:.6e} lin_t={grad_t:.6e} | "
            f"corr_abs_q_s={self._last_forward['corr_q_s']:.6e} | "
            f"broken={int(broken_win)} locked={int(locked_win)}"
        )
        if isinstance(self._last_attn_stats, dict):
            msg = (
                msg
                + " | "
                + f"attn_entropy mean={self._last_attn_stats.get('entropy_mean', float('nan')):.6e} "
                + f"attn_top1 mean={self._last_attn_stats.get('top1_mean', float('nan')):.6e} "
                + f"attn_margin mean={self._last_attn_stats.get('margin_mean', float('nan')):.6e}"
            )
        if logger_obj is not None:
            try:
                logger_obj.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

        if (broken_win or locked_win) and (self._last_warn_step != int(step_i)):
            self._last_warn_step = int(step_i)
            warn = (
                f"[wavelet condfilm_v2 alert] layer={lid} step={int(step_i)} "
                f"broken={int(broken_win)} locked={int(locked_win)} "
                f"sat_s_mean={lock_payload.get('sat_s_mean', float('nan')):.6e} "
                f"sat_t_mean={lock_payload.get('sat_t_mean', float('nan')):.6e} "
                f"grad_s_med={lock_payload.get('grad_s_med', float('nan')):.6e} "
                f"grad_t_med={lock_payload.get('grad_t_med', float('nan')):.6e}"
            )
            if logger_obj is not None:
                try:
                    logger_obj.warning(warn)
                except Exception:
                    print(warn)
            else:
                print(warn)


def build_wavelet_condfilm_v2_param_group(
    model: nn.Module,
    *,
    base_lr: float,
    lr_mult: float = 0.1,
    weight_decay: float = 0.0,
):
    params = [p for n, p in model.named_parameters() if ("wavelet_cond_film_v2" in n and p.requires_grad)]
    if len(params) == 0:
        return []
    return [
        {
            "params": params,
            "lr": float(base_lr) * float(lr_mult),
            "weight_decay": float(weight_decay),
        }
    ]

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

    For T > 4096, float32 back-substitution over 8192+ rows accumulates to overflow.
    Fix: use float64 for the triangular solve (batched across all heads at once).
    Peak memory: B×H×T²×8 bytes ≈ 8.6 GB for B=1,H=16,T=8192 — fits on 48 GB GPU.
    ~30x faster than head-by-head because cuBLAS can parallelize across the H dimension.
    """
    b0 = beta.to(compute_dtype)
    beta_h = b0.transpose(1, 2)  # [B,H,T] (column index j)

    T = A.shape[-1]
    if T > 16384:
        # Batched fp64 solve: all heads at once, cast back to compute_dtype after
        # (fp64 at T<=16384 peaks at 3x[T,T]@fp64; only use beyond 16384 to avoid OOM on 40GB GPU)
        Zt = torch.linalg.solve_triangular(
            A.to(torch.float64).transpose(-1, -2),
            S.to(torch.float64).transpose(-1, -2),
            upper=True,
            unitriangular=True,
        ).to(compute_dtype)
        Z = Zt.transpose(-1, -2)      # [B,H,T,T] = S @ A^{-1}
    else:
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
    drift_dampen_ltrain: int = 0,
    drift_dampen_alpha: float = 1.0,
):
    """
    returns:
      E_base_raw: [B,H,T,T]   lower(QK^T) - M_base@strictLower(WK^T), NO mask, NO scale
      M_base:     [B,H,T,T]
      strict_WK:  [B,H,T,T]
      A:          [B,H,T,T]
      lower_QK:   [B,H,T,T]   plain content-matching term (pre-subtraction)
      correction: [B,H,T,T]   M_base @ strict_WK, state-transition term (pre-subtraction,
                               post drift-dampen if requested)

    drift_dampen_ltrain/alpha: state-transition-drift test. M_base is the cumulative
    Householder-transition-derived correction coefficient; M_base @ strict_WK is the
    actual state-transition contribution to the logit (as opposed to lower_QK, the
    plain content-matching term). When ltrain>0 and alpha!=1.0, this dampens the
    transition CORRECTION term only (not the content term), for queries in the last
    L_train rows (q >= T-L_train) and keys reaching back further than the extrapolation
    overhang (dist > T-L_train) -- i.e. restricts the parameter matrix that encodes the
    transition itself, rather than post-hoc rescaling the already-combined logit.
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
    del QK
    strict_WK = torch.tril(WK, diagonal=-1)
    del WK

    QW = torch.einsum("b i h d, b j h d -> b h i j", q0, w0)
    S_base = torch.tril(QW, diagonal=0)
    del QW

    M_base = path_ut_M_from_S(A, S_base, b0, compute_dtype=compute_dtype)
    del S_base

    correction = M_base @ strict_WK
    if drift_dampen_ltrain > 0 and drift_dampen_alpha != 1.0:
        overhang = T - drift_dampen_ltrain
        if overhang > 0:
            q_pos = torch.arange(T, device=correction.device, dtype=torch.long).view(1, 1, -1, 1)
            k_pos = torch.arange(T, device=correction.device, dtype=torch.long).view(1, 1, 1, -1)
            last_ltrain_query_mask = q_pos >= overhang
            dist = q_pos - k_pos
            beyond_overhang_mask = dist > overhang
            drift_mask = last_ltrain_query_mask & beyond_overhang_mask
            correction = torch.where(drift_mask, correction * drift_dampen_alpha, correction)

    E_base_raw = lower_QK - correction
    return E_base_raw, M_base, strict_WK, A, lower_QK, correction

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
        wavelet_mode: str = "additive",   # off | router_rel | logit_bias | logit_bias_ctxscale_shift_v0 | logit_bias_ctxscale_shift_v0_film | cond_film_v2
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
        self.bias_type = getattr(config, "bias_type", "wavelet")
        self.wavelet_ctxscale_pattern_mode = str(
            getattr(config, "wavelet_ctxscale_pattern_mode", getattr(config, "pattern_mode", "ricker"))
        ).strip().lower()
        if self.wavelet_ctxscale_pattern_mode not in ("ricker", "pl4", "restore"):
            raise ValueError(
                "wavelet_ctxscale_pattern_mode/pattern_mode must be one of "
                f"'ricker', 'pl4', or 'restore', got {self.wavelet_ctxscale_pattern_mode!r}"
            )
        self.wavelet_ctxscale_restore_bin = int(
            getattr(config, "wavelet_ctxscale_restore_bin", getattr(config, "restore_bin", -1))
        )
        if self.wavelet_ctxscale_pattern_mode == "restore" and not (0 <= self.wavelet_ctxscale_restore_bin < 4):
            raise ValueError(
                "wavelet_ctxscale_restore_bin/restore_bin must be in [0, 3] "
                f"when pattern_mode='restore', got {self.wavelet_ctxscale_restore_bin}"
            )
        if self.wavelet_ctxscale_pattern_mode != "ricker" and self.bias_type != "wavelet":
            raise ValueError(
                "PL4/restoration pattern ablations are defined only for bias_type='wavelet', "
                f"got bias_type={self.bias_type!r}"
            )
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
        self.attn_norm = str(getattr(config, "attn_norm", "softmax")).strip().lower()
        self.entmax_alpha = float(getattr(config, "entmax_alpha", 1.5))
        self.entmax_scope = str(getattr(config, "entmax_scope", "all")).strip().lower()
        self.entmax_layers = self._parse_layer_set(getattr(config, "entmax_layers", "8,9,10,11"))
        self.entmax_stable_heads_csv = getattr(config, "entmax_stable_heads_csv", None)
        self.entmax_stable_heads = self._load_head_pair_csv(self.entmax_stable_heads_csv)

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
        self.path_attn_impl = self._normalize_path_attn_impl(
            getattr(config, "path_attn_impl", "pytorch")
        )
        self.use_wavelet_beta = use_wavelet_beta
        self.wavelet_mode = wavelet_mode
        self.wavelet_mode_resolved = self._normalize_wavelet_mode(getattr(config, "wavelet_mode", wavelet_mode))
        self.wavelet_logit_bias_eps = float(getattr(config, "wavelet_logit_bias_eps", 1e-6))
        self.wavelet_logit_bias_debug_assert = self._as_bool(
            getattr(config, "wavelet_logit_bias_debug_assert", False), default=False
        )
        self.wavelet_logit_bias_eval_mult = float(getattr(config, "wavelet_logit_bias_eval_mult", 1.0))
        self.wavelet_ctxscale_gain_st = self._as_bool(
            getattr(config, "wavelet_ctxscale_gain_st", False), default=False
        )
        self._last_pa_raw_logits_unconditional = None
        self._last_pa_lower_QK = None
        self._last_pa_correction = None
        # Opt-in only: storing these keeps 2 extra [B,H,T,T] float32 tensors alive per
        # layer past when the forward pass would otherwise free them (they're consumed
        # into E_base_raw = lower_QK - correction). At L4096 that's ~1.5GiB each --
        # enough to OOM eval jobs that don't need them. Only the interference-probe
        # scripts (PAT-251) should turn this on.
        self.wavelet_pa_debug_store_correction_terms = self._as_bool(
            getattr(config, "wavelet_pa_debug_store_correction_terms", False), default=False
        )
        self.wavelet_pa_beyond_dampen_threshold = int(
            getattr(config, "wavelet_pa_beyond_dampen_threshold", 0)
        )
        self.wavelet_pa_beyond_dampen_alpha = float(
            getattr(config, "wavelet_pa_beyond_dampen_alpha", 1.0)
        )
        self.wavelet_qwab_beyond_dampen_threshold = int(
            getattr(config, "wavelet_qwab_beyond_dampen_threshold", 0)
        )
        self.wavelet_qwab_beyond_dampen_alpha = float(
            getattr(config, "wavelet_qwab_beyond_dampen_alpha", 1.0)
        )
        self.wavelet_qwab_within_dampen_threshold = int(
            getattr(config, "wavelet_qwab_within_dampen_threshold", 0)
        )
        self.wavelet_qwab_within_dampen_alpha = float(
            getattr(config, "wavelet_qwab_within_dampen_alpha", 1.0)
        )
        self.wavelet_pa_within_dampen_threshold = int(
            getattr(config, "wavelet_pa_within_dampen_threshold", 0)
        )
        self.wavelet_pa_within_dampen_alpha = float(
            getattr(config, "wavelet_pa_within_dampen_alpha", 1.0)
        )
        # State-transition-drift test: for queries in the LAST L_train rows only
        # (the ones that have accumulated the full T-L_train extrapolation overhang),
        # dampen keys reaching back further than that same overhang (T-L_train).
        # Both the query scope and the distance threshold derive from L_train alone.
        self.wavelet_pa_state_drift_dampen_ltrain = int(
            getattr(config, "wavelet_pa_state_drift_dampen_ltrain", 0)
        )
        self.wavelet_pa_state_drift_dampen_alpha = float(
            getattr(config, "wavelet_pa_state_drift_dampen_alpha", 1.0)
        )
        self.wavelet_logit_bias_rms_scope = str(
            getattr(config, "wavelet_logit_bias_rms_scope", "context")
        ).strip().lower()
        if self.wavelet_logit_bias_rms_scope in ("full", "all", "all_context", "context_length"):
            self.wavelet_logit_bias_rms_scope = "context"
        if self.wavelet_logit_bias_rms_scope == "causal":
            raise ValueError(
                "wavelet_logit_bias_rms_scope='causal' is disabled for PAT-234: "
                "causal RMS changes the waveform scale row-by-row. Use 'context'."
            )
        if self.wavelet_logit_bias_rms_scope != "context":
            raise ValueError(
                "wavelet_logit_bias_rms_scope must be 'context', "
                f"got {self.wavelet_logit_bias_rms_scope!r}"
            )
        # PAT-234 variant C: center the per-scale basis over causal keys before RMS-norm,
        # removing the softmax-invisible key-independent (DC) component so coarse
        # (near-constant) scales can actually influence attention. Default off.
        self.wavelet_logit_bias_center = self._as_bool(
            getattr(config, "wavelet_logit_bias_center_enable", False), default=False
        )
        # PAT-234: disable the per-scale RMS-norm entirely -> raw (absolute) wavelet basis.
        # This is the only length/position-invariant normalization choice (all-T RMS is
        # length-dependent, causal RMS is position-dependent / OOD past train length).
        # Cost: scales no longer equalized (coarse=DC-invisible, fine=small spike). Default off.
        self.wavelet_logit_bias_norm_disable = self._as_bool(
            getattr(config, "wavelet_logit_bias_norm_disable", False), default=False
        )
        self.wavelet_logit_bias_log_sample_tokens = max(
            1, int(getattr(config, "wavelet_logit_bias_log_sample_tokens", 64))
        )
        self.wavelet_logit_bias_log_sample_heads = max(
            1, int(getattr(config, "wavelet_logit_bias_log_sample_heads", 4))
        )
        self.wavelet_logit_bias_local_step = 0
        # PAT-225: scale cardinality S is configurable; default 8 keeps every
        # pre-existing config/checkpoint bit-identical (router_band_num does NOT
        # control this branch — it only feeds the legacy router1/router2 modes).
        self.wavelet_ctxscale_k = max(1, int(getattr(config, "wavelet_ctxscale_k", 8)))
        # PAT-225 mechanism probe: inference-time per-scale knockout. Comma-
        # separated 0-based scale indices whose router logit is forced to -1e4
        # (atom off in every router mode). Empty/absent = no-op.
        _mask_raw = str(getattr(config, "wavelet_ctxscale_scale_mask", "") or "").strip()
        self.wavelet_ctxscale_scale_mask_idx = tuple(
            int(x) for x in _mask_raw.split(",") if x.strip() != ""
        ) if _mask_raw else ()
        if self.wavelet_ctxscale_scale_mask_idx and layer_idx in (0, None):
            print(
                f"[PAT-225] scale-knockout active: masking scale indices "
                f"{list(self.wavelet_ctxscale_scale_mask_idx)} of K={self.wavelet_ctxscale_k}",
                flush=True,
            )
        # PAT-225 seed-variance probe: inference-time per-LAYER knockout. Comma-
        # separated 0-based layer indices whose null-vs-nonnull gate logit is forced
        # to -1e4 (g0_gate->0, pi_null->1), i.e. that layer's wavelet bias is fully
        # disabled and it degenerates to baseline PaTH attention. Empty/absent = no-op.
        _ko_layers_raw = str(getattr(config, "wavelet_ctxscale_ko_layers", "") or "").strip()
        self.wavelet_ctxscale_ko_layers_idx = tuple(
            int(x) for x in _ko_layers_raw.split(",") if x.strip() != ""
        ) if _ko_layers_raw else ()
        _ko_query_raw = str(
            getattr(config, "wavelet_ctxscale_ko_query_ranges", "") or ""
        ).strip()
        _ko_query_ranges = []
        if _ko_query_raw:
            for _item in re.split(r"[;,]", _ko_query_raw):
                _item = _item.strip()
                if not _item:
                    continue
                if ":" not in _item:
                    raise ValueError(
                        "wavelet_ctxscale_ko_query_ranges must contain half-open "
                        f"start:end ranges, got {_item!r}."
                    )
                _start_raw, _end_raw = _item.split(":", 1)
                _start, _end = int(_start_raw), int(_end_raw)
                if _start < 0 or _end <= _start:
                    raise ValueError(
                        "wavelet_ctxscale_ko_query_ranges requires 0 <= start < end, "
                        f"got {_item!r}."
                    )
                _ko_query_ranges.append((_start, _end))
        self.wavelet_ctxscale_ko_query_ranges_idx = tuple(_ko_query_ranges)
        if self.wavelet_ctxscale_ko_layers_idx and layer_idx in (0, None):
            print(
                f"[PAT-225] layer-knockout active: forcing layers "
                f"{list(self.wavelet_ctxscale_ko_layers_idx)} to null (wavelet bias off); "
                f"query_ranges={list(self.wavelet_ctxscale_ko_query_ranges_idx) or 'all'}",
                flush=True,
            )
        qwab_groups_per_layer = max(1, int(getattr(config, "qwab_groups_per_layer", 1)))
        if qwab_groups_per_layer != 1:
            raise ValueError(
                "Head-wise wavelet routing has been removed; "
                "qwab_groups_per_layer must be 1."
            )
        self.wavelet_ctxscale_tau = float(getattr(config, "wavelet_ctxscale_tau", getattr(config, "tau", 1.0)))
        # PAT-225: fixed (non-learnable) bandwidth for router_sigmoid_mode="gaussian_kernel"
        # -- a single per-query "focus" scalar mapped through a FIXED unimodal Gaussian
        # kernel in log2(scale) space, as opposed to K independent per-scale weights
        # (signed/positive). Per-query DOF = 2 (g0 + focus), matching K1's DOF=1 plus
        # exactly one new knob. sigma stays a fixed hyperparameter (not learned, not
        # per-query) so it doesn't add a third per-query degree of freedom.
        self.wavelet_ctxscale_kernel_sigma = float(
            getattr(config, "wavelet_ctxscale_kernel_sigma", 0.5)
        )
        self.wavelet_ctxscale_tau_schedule = str(
            getattr(config, "wavelet_ctxscale_tau_schedule", "none")
        ).strip().lower()
        if self.wavelet_ctxscale_tau_schedule not in ("none", "linear"):
            self.wavelet_ctxscale_tau_schedule = "none"
        self.wavelet_ctxscale_tau_start = float(
            getattr(config, "wavelet_ctxscale_tau_start", self.wavelet_ctxscale_tau)
        )
        self.wavelet_ctxscale_tau_end = float(
            getattr(config, "wavelet_ctxscale_tau_end", self.wavelet_ctxscale_tau)
        )
        self.wavelet_ctxscale_tau_anneal_steps = int(
            getattr(config, "wavelet_ctxscale_tau_anneal_steps", 0)
        )
        self.wavelet_ctxscale_tau_anneal_warmup = int(
            getattr(config, "wavelet_ctxscale_tau_anneal_warmup", 0)
        )
        self._last_router_jitter_stats = {}
        self._last_router_entropy_reg_loss = torch.tensor(0.0)
        self._last_router_entropy_reg_active_frac = torch.tensor(0.0)
        self.wavelet_ctxscale_rho_override = getattr(config, "wavelet_ctxscale_rho_override", None)
        self.wavelet_ctxscale_router_rms_eps = float(getattr(config, "wavelet_ctxscale_router_rms_eps", 1e-6))
        # PAT-244: router-logit normalization mode + decoupled learnable temperatures.
        #   "none"                 -> current behavior (single shared tau, no norm)
        #   "rms_joint"            -> re-add the removed joint RMS over [null, scales] (baseline-rms)
        #   "dual_temp"            -> tau_null on g0; tau_scale on scale branch (with_null: sigmoid-normalize w=g/sum(g); independent: sigmoid). SOFTMAX FORBIDDEN.
        #   "dual_temp_scale_rms"  -> as dual_temp, but RMS-norm the scale logits (null excluded) first
        #   "dual_temp_scale_none" -> tau_null on g0; scale branch has no temperature (raw)
        self.wavelet_router_norm_mode = str(
            getattr(config, "wavelet_router_norm_mode", "none")
        ).strip().lower()
        if self.wavelet_router_norm_mode not in (
            "none", "rms_joint", "dual_temp", "dual_temp_scale_rms", "dual_temp_scale_none"
        ):
            self.wavelet_router_norm_mode = "none"
        # PAT-244 unified cosine router: L2-normalize router feature + each weight row so
        # every logit (null + scales) is a cosine in [-1,1]. Composes with norm_mode:
        # cosine + dual_temp = the well-posed learnable-temperature design; cosine + none
        # = fixed cosine router. Default off (raw logits).
        self.wavelet_router_cosine = self._as_bool(getattr(config, "wavelet_router_cosine", False), default=False)
        if self.wavelet_router_norm_mode in ("dual_temp", "dual_temp_scale_rms", "dual_temp_scale_none"):
            # Bounded log-sigmoid temperature: tau = tau_min * (tau_max/tau_min)^sigmoid(raw).
            # Keeps tau in [tau_min, tau_max] with a smooth (never-zero) gradient, preventing
            # the unbounded-softplus instability where per-layer temps diverged to ~1e31 or
            # collapsed to ~1e-4. Init raw via the inverse map so tau starts at tau_*_init.
            self.router_tau_min = float(getattr(config, "wavelet_router_tau_min", 0.1))
            self.router_tau_max = float(getattr(config, "wavelet_router_tau_max", 10.0))
            _tau_null_init = float(getattr(config, "wavelet_router_tau_null_init", 1.0))
            _tau_scale_init = float(getattr(config, "wavelet_router_tau_scale_init", 1.0))
            def _inv_bounded_tau(t):
                lo, hi = self.router_tau_min, self.router_tau_max
                t = min(max(float(t), lo * (1.0 + 1e-4)), hi * (1.0 - 1e-4))
                frac = math.log(t / lo) / math.log(hi / lo)
                frac = min(max(frac, 1e-4), 1.0 - 1e-4)
                return math.log(frac / (1.0 - frac))  # logit
            self.router_tau_null_raw = nn.Parameter(
                torch.tensor(_inv_bounded_tau(_tau_null_init), dtype=torch.float32)
            )
            self.router_tau_scale_raw = nn.Parameter(
                torch.tensor(_inv_bounded_tau(_tau_scale_init), dtype=torch.float32)
            )
        self.wavelet_ctxscale_chunk_q = max(1, int(getattr(config, "wavelet_ctxscale_chunk_q", 128)))
        self.wavelet_ctxscale_max_log_samples = max(
            128, int(getattr(config, "wavelet_ctxscale_max_log_samples", 4096))
        )
        self.wavelet_ctx_feat_mode = str(
            getattr(config, "wavelet_ctx_feat_mode", "q_meanH")
        ).strip()
        if self.wavelet_ctx_feat_mode.lower() in (
            "q_perh",
            "q_headwise",
            "q_hw",
            "q_minus_qcorr_perh",
            "q_minus_qcorr_headwise",
            "dq_perh",
        ):
            raise ValueError(
                "Head-wise wavelet routing has been removed; "
                f"wavelet_ctx_feat_mode={self.wavelet_ctx_feat_mode!r} is unsupported."
            )
        self.wavelet_ctx_feat_rms_eps = float(getattr(config, "wavelet_ctx_feat_rms_eps", 1e-6))
        self.wavelet_ctx_feat_detach_delta = self._as_bool(
            getattr(config, "wavelet_ctx_feat_detach_delta", False), default=False
        )
        self.wavelet_ctxscale_g_max = float(getattr(config, "wavelet_ctxscale_g_max", 0.5))
        self.wavelet_ctxscale_g_bias_max = float(getattr(config, "wavelet_ctxscale_g_bias_max", 4.0))
        self.wavelet_ctxscale_disable_layer_gate = self._as_bool(
            getattr(config, "wavelet_ctxscale_disable_layer_gate", False), default=False
        )
        self.wavelet_ctxscale_use_head_gate = self._as_bool(
            getattr(config, "wavelet_ctxscale_use_head_gate", False), default=False
        )
        self.wavelet_ctxscale_scale_dependent_shift = self._as_bool(
            getattr(config, "wavelet_ctxscale_scale_dependent_shift", False), default=False
        )
        self.wavelet_ctxscale_shift_legacy_symmetric = self._as_bool(
            getattr(config, "wavelet_ctxscale_shift_legacy_symmetric", False), default=False
        )
        # PAT-244: unconditional-RMS mode -- bypass the router entirely (no null gate,
        # no scale selection). The single wavelet basis (K must be 1) is still built,
        # RMS-normalized, and shift-applied exactly as usual, but the per-token routing
        # weight that normally multiplies it (pi_scale, incorporating the null gate) is
        # replaced with a constant 1.0, so the bias is always added at full strength.
        # wavelet_ctx_router is still constructed but receives no gradient in this mode.
        self.wavelet_ctxscale_unconditional_rms = self._as_bool(
            getattr(config, "wavelet_ctxscale_unconditional_rms", False), default=False
        )
        self.wavelet_ctxscale_fixed_scale_ratio = getattr(
            config, "wavelet_ctxscale_fixed_scale_ratio", None
        )
        # PAT-225: query-INDEPENDENT but LEARNED per-scale ratio -- a single nn.Parameter
        # per layer, shared across every query/token (not derived from router_logits at
        # all), optimized by gradient descent like any other weight. Sits between
        # "fixed_scale_ratio" (hand-set constant, never learned) and
        # "with_null_independent_scales" (learned AND query-conditioned): isolates
        # whether query-conditioning itself matters, vs. just learning a better-than-1/3
        # static mixture. Mutually exclusive with wavelet_ctxscale_fixed_scale_ratio.
        self.wavelet_ctxscale_ratio_learnable = self._as_bool(
            getattr(config, "wavelet_ctxscale_ratio_learnable", False), default=False
        )
        if self.wavelet_ctxscale_ratio_learnable and self.wavelet_ctxscale_fixed_scale_ratio is not None:
            raise ValueError(
                "wavelet_ctxscale_ratio_learnable and wavelet_ctxscale_fixed_scale_ratio "
                "are mutually exclusive."
            )
        # PAT-253: query-INDEPENDENT but LEARNED null/apply gate -- single nn.Parameter
        # per layer, shared across every query/token, replacing g0_gate's dependence on
        # router_logits[...,0:1] entirely. Plays the same structural role for the
        # null/apply decision that wavelet_ctxscale_ratio_learnable plays for the
        # scale-mixture: isolates whether query-conditioning of *this* decision matters,
        # independent of whatever scale-mixture mode (default/fixed/ratio_learnable) is
        # active. Only meaningful under wavelet_router_sigmoid_mode=
        # "with_null_independent_scales" (enforced at use-site).
        self.wavelet_ctxscale_g0_learnable = self._as_bool(
            getattr(config, "wavelet_ctxscale_g0_learnable", False), default=False
        )
        # PAT-253: hand-set constant null/apply gate -- e.g. 1.0 means "always fully
        # apply the wavelet bias branch, no gating at all". Zero learnable parameters
        # for this decision (unlike g0_learnable's per-layer nn.Parameter), the true
        # "everything fixed" endpoint paired with fixed_scale_ratio, so a
        # fixed_scale_ratio + g0_fixed_value=1.0 config has NO learned routing
        # parameters whatsoever. Mutually exclusive with g0_learnable.
        self.wavelet_ctxscale_g0_fixed_value = getattr(config, "wavelet_ctxscale_g0_fixed_value", None)
        if self.wavelet_ctxscale_g0_fixed_value is not None:
            self.wavelet_ctxscale_g0_fixed_value = float(self.wavelet_ctxscale_g0_fixed_value)
            if not (0.0 <= self.wavelet_ctxscale_g0_fixed_value <= 1.0):
                raise ValueError(
                    "wavelet_ctxscale_g0_fixed_value must be in [0,1], got "
                    f"{self.wavelet_ctxscale_g0_fixed_value}."
                )
        if self.wavelet_ctxscale_g0_learnable and self.wavelet_ctxscale_g0_fixed_value is not None:
            raise ValueError(
                "wavelet_ctxscale_g0_learnable and wavelet_ctxscale_g0_fixed_value "
                "are mutually exclusive."
            )
        # PAT-244: opt-in independent per-scale shift head. Default (False) keeps the
        # original single shared shift_proj (1 output, same beta_m applied to every
        # scale index, just rescaled by each scale's own rho_i under
        # scale_dependent_shift=true, or literally identical under
        # scale_dependent_shift=false). True gives each of the K scales its own
        # learned shift decision (shift_proj outputs K values instead of 1) --
        # required for a true "same scale, independently-learned shift" test.
        # Changes wavelet_shift_proj's output shape, so checkpoints trained with
        # this flag are NOT loadable under the default (and vice versa).
        self.wavelet_ctxscale_shift_per_scale = self._as_bool(
            getattr(config, "wavelet_ctxscale_shift_per_scale", False), default=False
        )
        # PAT-244: shift_number (S) generates S independently-learnable-shift
        # copies of EACH distinct scale in wavelet_ctxscale_scale_max_exp, while
        # the router still makes only ONE weighting decision per distinct scale
        # (all S copies of a given scale share the same router gate). This is
        # architecturally different from just listing the same scale K times in
        # wavelet_ctxscale_scale_max_exp, which forces the router to separately
        # (and redundantly) learn to weight near-identical candidates. Requires
        # wavelet_ctxscale_shift_per_scale=true (otherwise the S copies of a
        # scale would share not just the router gate but also the shift decision,
        # making them literally identical). Default 1 is a no-op, bit-identical
        # to not having this flag at all.
        self.wavelet_ctxscale_shift_number = int(getattr(config, "wavelet_ctxscale_shift_number", 1))
        if self.wavelet_ctxscale_shift_number < 1:
            raise ValueError(
                f"wavelet_ctxscale_shift_number must be >= 1, got {self.wavelet_ctxscale_shift_number}"
            )
        if self.wavelet_ctxscale_shift_number > 1 and not self.wavelet_ctxscale_shift_per_scale:
            raise ValueError(
                "wavelet_ctxscale_shift_number > 1 requires wavelet_ctxscale_shift_per_scale=true "
                "(otherwise the shift-number copies of a scale would share the same shift decision "
                "too, making them literally identical, not just router-tied)."
            )
        self.wavelet_ctxscale_k_total = self.wavelet_ctxscale_k * self.wavelet_ctxscale_shift_number
        # PAT-225: cap RMS-statistics window during eval; 0 keeps prior full-width behavior.
        self.wavelet_ctxscale_rms_train_window = int(
            getattr(config, "wavelet_ctxscale_rms_train_window", 0)
        )
        if self.wavelet_ctxscale_rms_train_window < 0:
            raise ValueError(
                f"wavelet_ctxscale_rms_train_window must be >= 0, got {self.wavelet_ctxscale_rms_train_window}"
            )
        if self._as_bool(getattr(config, "lw_residual_hw_enable", False), default=False):
            raise ValueError(
                "Head-wise wavelet routing has been removed; "
                "lw_residual_hw_enable must be false."
            )
        self.wavelet_ctxscale_use_relative_position = self._as_bool(
            getattr(config, "wavelet_ctxscale_use_relative_position", False), default=False
        )
        # In key-anchor mode (use_relative_position=False), shift wavelet u-origin from
        # absolute key position 0 to query-dependent center: center = ratio * query_pos.
        self.wavelet_ctxscale_center_pos_ratio = float(
            getattr(config, "wavelet_ctxscale_center_pos_ratio", 0.0)
        )
        if not math.isfinite(self.wavelet_ctxscale_center_pos_ratio):
            self.wavelet_ctxscale_center_pos_ratio = 0.0
        self.wavelet_ctxscale_center_pos_ratio = max(
            0.0, min(1.0, self.wavelet_ctxscale_center_pos_ratio)
        )
        # PAT-234 dual-center ablation: one per-scale waveform is the sum of the
        # absolute-key Ricker pattern and the query-centered Ricker pattern, then
        # the existing per-scale centering/RMS is applied to the summed waveform.
        self.wavelet_ctxscale_dual_center_enable = self._as_bool(
            getattr(
                config,
                "wavelet_ctxscale_dual_center_enable",
                getattr(config, "wavelet_ctxscale_dual_center", False),
            ),
            default=False,
        )
        self.wavelet_ctxscale_dual_center_norm_mode = str(
            getattr(config, "wavelet_ctxscale_dual_center_norm_mode", "sum_then_rms")
        ).strip().lower()
        if self.wavelet_ctxscale_dual_center_norm_mode in ("sum", "summed"):
            self.wavelet_ctxscale_dual_center_norm_mode = "sum_then_rms"
        if self.wavelet_ctxscale_dual_center_norm_mode in (
            "separate",
            "separate_norm",
            "separate_rms_sqrt2",
        ):
            self.wavelet_ctxscale_dual_center_norm_mode = "separate_rms"
        if self.wavelet_ctxscale_dual_center_norm_mode not in (
            "sum_then_rms",
            "separate_rms",
            "separate_rms_nosqrt",
        ):
            raise ValueError(
                "wavelet_ctxscale_dual_center_norm_mode must be 'sum_then_rms', "
                f"'separate_rms', or 'separate_rms_nosqrt', got {self.wavelet_ctxscale_dual_center_norm_mode!r}"
            )
        if self.wavelet_ctxscale_dual_center_enable:
            if self.bias_type != "wavelet" or self.wavelet_ctxscale_pattern_mode != "ricker":
                raise ValueError(
                    "wavelet_ctxscale_dual_center_enable requires bias_type='wavelet' "
                    "and wavelet_ctxscale_pattern_mode='ricker'."
                )
            if self.wavelet_ctxscale_use_relative_position:
                raise ValueError(
                    "wavelet_ctxscale_dual_center_enable is incompatible with "
                    "wavelet_ctxscale_use_relative_position; it explicitly builds "
                    "absolute and query-centered coordinates."
                )
        if (
            self.wavelet_ctxscale_pattern_mode != "ricker"
            and (
                self.wavelet_ctxscale_dual_center_enable
                or self.wavelet_ctxscale_use_relative_position
                or self.wavelet_ctxscale_center_pos_ratio > 0.0
            )
        ):
            raise ValueError(
                "PL4/restoration pattern ablations are only supported for the "
                "absolute center-0 coordinate."
            )
        # E3 ablation: learnable global anchor offset for wavelet position reference
        self.wavelet_ctxscale_learnable_anchor = self._as_bool(
            getattr(config, "wavelet_ctxscale_learnable_anchor", False), default=False
        )
        if self.wavelet_ctxscale_learnable_anchor:
            self.wavelet_anchor_offset = nn.Parameter(torch.tensor(0.0))
        else:
            self.wavelet_anchor_offset = None
        self.wavelet_ctxscale_shift_unit_max = float(getattr(config, "wavelet_ctxscale_shift_unit_max", 1.0))
        # PAT-244: morlet's internal oscillation frequency (cos(freq*u) inside the
        # envelope) was hardcoded at 5.0; exposed as a config knob for a frequency sweep.
        self.wavelet_morlet_freq = float(getattr(config, "wavelet_morlet_freq", 5.0))
        self.wavelet_shift_T_mode = str(getattr(config, "wavelet_shift_T_mode", "legacy")).strip().lower()
        if self.wavelet_shift_T_mode not in ("legacy", "runtime", "train_ref"):
            self.wavelet_shift_T_mode = "legacy"
        self.wavelet_shift_T_ref = max(2, int(getattr(config, "wavelet_shift_T_ref", 512)))
        self.wavelet_basis_control = str(getattr(config, "wavelet_basis_control", "none")).strip().lower()
        if self.wavelet_basis_control not in ("none", "permute_scales", "random_basis"):
            self.wavelet_basis_control = "none"
        if self.wavelet_ctxscale_pattern_mode != "ricker" and self.wavelet_basis_control == "random_basis":
            raise ValueError(
                "PL4/restoration pattern ablations require the generated Ricker basis; "
                "wavelet_basis_control='random_basis' is incompatible."
            )
        self.wavelet_router_sigmoid_mode = str(getattr(config, "wavelet_router_sigmoid_mode", "softmax")).strip().lower()
        if self.wavelet_router_sigmoid_mode not in ("softmax", "with_null", "no_null", "with_null_independent_scales", "signed"):
            self.wavelet_router_sigmoid_mode = "softmax"
        # Optional eval-time wavelet intervention hook (default off).
        self.wavelet_intervention_enable = self._as_bool(
            getattr(config, "wavelet_intervention_enable", False), default=False
        )
        self.wavelet_intervention_strict = self._as_bool(
            getattr(config, "wavelet_intervention_strict", True), default=True
        )
        self.wavelet_intervention_mode = str(
            getattr(config, "wavelet_intervention_mode", "ctxscale_null")
        ).strip().lower()
        self._wavelet_intervention_targets = self._parse_wavelet_intervention_targets(
            getattr(config, "wavelet_intervention_targets", None),
            default_layer=getattr(config, "wavelet_intervention_layer", None),
            default_heads=getattr(config, "wavelet_intervention_heads", None),
        )
        self._wavelet_basis_seed_warned = False
        self._wavelet_basis_perm_cache = {}
        self._wavelet_basis_random_cache = {}
        self.wavelet_viz_export = self._as_bool(getattr(config, "wavelet_viz_export", False), default=False)
        self.wavelet_viz_max_batches = max(1, int(getattr(config, "wavelet_viz_max_batches", 8)))
        self.wavelet_viz_sample_q = max(
            1,
            int(
                getattr(
                    config,
                    "wavelet_viz_sample_q",
                    getattr(config, "wavelet_logit_bias_log_sample_tokens", 64),
                )
            ),
        )
        self.wavelet_viz_sample_k = max(1, int(getattr(config, "wavelet_viz_sample_k", 256)))
        self.wavelet_viz_outdir = str(
            getattr(config, "wavelet_viz_outdir", getattr(config, "wavelet_analysis_output_dir", ""))
        ).strip()
        self.wavelet_viz_run_tag = str(getattr(config, "wavelet_viz_run_tag", "default")).strip() or "default"
        self.wavelet_viz_mode = str(
            getattr(config, "wavelet_viz_mode", getattr(config, "wavelet_mode", "unknown"))
        ).strip()
        self.wavelet_viz_model_size = str(getattr(config, "wavelet_viz_model_size", "unknown")).strip() or "unknown"
        self.wavelet_viz_seed = int(getattr(config, "wavelet_viz_seed", getattr(config, "seed", -1)))
        self.wavelet_analysis_export = self._as_bool(getattr(config, "wavelet_analysis_export", False), default=False)
        self.wavelet_analysis_max_q = max(
            1,
            int(
                getattr(
                    config,
                    "wavelet_analysis_max_q",
                    getattr(config, "wavelet_logit_bias_log_sample_tokens", 64),
                )
            ),
        )
        self.wavelet_analysis_max_batches = max(1, int(getattr(config, "wavelet_analysis_max_batches", 64)))
        self.wavelet_analysis_output_dir = str(
            getattr(config, "wavelet_analysis_output_dir", getattr(config, "save_root", ""))
        ).strip()
        self.wavelet_analysis_run_tag = str(getattr(config, "wavelet_analysis_run_tag", "default")).strip() or "default"
        self.wavelet_analysis_mode = str(
            getattr(config, "wavelet_analysis_mode", getattr(config, "wavelet_mode", "unknown"))
        ).strip()
        self.wavelet_analysis_seed = int(getattr(config, "wavelet_analysis_seed", getattr(config, "seed", -1)))
        self._wavelet_analysis_layer_counts = {}
        self._wavelet_analysis_layer_steps = {}
        self._wavelet_analysis_warned = set()
        # Eval-only attention heatmap export (for per-layer/per-head comparison).
        self.eval_attn_heatmap_enabled = self._as_bool(
            getattr(config, "eval_attn_heatmap_enabled", False), default=False
        )
        self.eval_attn_heatmap_layers = self._parse_rel_layer_set(
            getattr(config, "eval_attn_heatmap_layers", "all")
        )
        self.eval_attn_heatmap_case_limit = max(
            1, int(getattr(config, "eval_attn_heatmap_case_limit", 1))
        )
        self.eval_attn_heatmap_case_index = max(
            0, int(getattr(config, "eval_attn_heatmap_case_index", 0))
        )
        self.eval_attn_heatmap_head_limit = max(
            0, int(getattr(config, "eval_attn_heatmap_head_limit", 0))
        )
        self.eval_attn_heatmap_max_seq = max(
            0, int(getattr(config, "eval_attn_heatmap_max_seq", 0))
        )
        self.eval_attn_heatmap_outdir = str(
            getattr(config, "eval_attn_heatmap_outdir", getattr(config, "save_root", "analysis"))
        ).strip() or "analysis"
        self.eval_attn_heatmap_run_tag = str(
            getattr(config, "eval_attn_heatmap_run_tag", "default")
        ).strip() or "default"
        self.eval_attn_heatmap_separate_step = self._as_bool(
            getattr(config, "eval_attn_heatmap_separate_step", False), default=False
        )
        self.eval_attn_heatmap_dpi = max(
            40, int(getattr(config, "eval_attn_heatmap_dpi", 80))
        )
        self.eval_attn_heatmap_save_png = self._as_bool(
            getattr(config, "eval_attn_heatmap_save_png", True), default=True
        )
        self.eval_attn_heatmap_save_pt = self._as_bool(
            getattr(config, "eval_attn_heatmap_save_pt", False), default=False
        )
        self.eval_attn_heatmap_save_pt_reduced_only = self._as_bool(
            getattr(config, "eval_attn_heatmap_save_pt_reduced_only", False), default=False
        )
        self.eval_attn_heatmap_pt_resize_to = max(
            0, int(getattr(config, "eval_attn_heatmap_pt_resize_to", 0))
        )
        # Keep .pt lightweight by default: save softmax maps only unless explicitly requested.
        self.eval_attn_heatmap_save_pt_logits = self._as_bool(
            getattr(config, "eval_attn_heatmap_save_pt_logits", False), default=False
        )
        self.eval_attn_heatmap_save_pt_outputs = self._as_bool(
            getattr(config, "eval_attn_heatmap_save_pt_outputs", False), default=False
        )
        self.eval_attn_heatmap_only_rel_layers = self._as_bool(
            getattr(config, "eval_attn_heatmap_only_rel_layers", False), default=False
        )
        self.eval_attn_heatmap_cmap = str(
            getattr(config, "eval_attn_heatmap_cmap", "viridis")
        ).strip() or "viridis"
        self.eval_attn_heatmap_delta_cmap = str(
            getattr(config, "eval_attn_heatmap_delta_cmap", "coolwarm")
        ).strip() or "coolwarm"
        self.eval_attn_heatmap_vmax_quantile = float(
            getattr(config, "eval_attn_heatmap_vmax_quantile", 0.999)
        )
        self.eval_attn_heatmap_delta_quantile = float(
            getattr(config, "eval_attn_heatmap_delta_quantile", 0.999)
        )
        self.eval_attn_heatmap_logit_delta_png = self._as_bool(
            getattr(config, "eval_attn_heatmap_logit_delta_png", True), default=True
        )
        self.eval_attn_heatmap_logit_delta_cmap = str(
            getattr(config, "eval_attn_heatmap_logit_delta_cmap", "coolwarm")
        ).strip() or "coolwarm"
        self.eval_attn_heatmap_logit_delta_quantile = float(
            getattr(config, "eval_attn_heatmap_logit_delta_quantile", self.eval_attn_heatmap_delta_quantile)
        )
        self.eval_attn_heatmap_show_colorbar = self._as_bool(
            getattr(config, "eval_attn_heatmap_show_colorbar", True), default=True
        )
        self.eval_attn_heatmap_min_valid_keys = max(
            1, int(getattr(config, "eval_attn_heatmap_min_valid_keys", 64))
        )
        self.eval_attn_heatmap_xtick_stride = max(
            0, int(getattr(config, "eval_attn_heatmap_xtick_stride", 512))
        )
        self.eval_attn_heatmap_stop_after_case = self._as_bool(
            getattr(config, "eval_attn_heatmap_stop_after_case", False), default=False
        )
        self.eval_attn_heatmap_keep_counter = self._as_bool(
            getattr(config, "eval_attn_heatmap_keep_counter", False), default=False
        )
        self.eval_attn_heatmap_stop_layer = self._to_int_or_none(
            getattr(config, "eval_attn_heatmap_stop_layer", None)
        )
        # Optional top-k token table export for each attention row (eval-only).
        self.eval_attn_topk_enabled = self._as_bool(
            getattr(config, "eval_attn_topk_enabled", True), default=True
        )
        self.eval_attn_topk_k = max(
            1, int(getattr(config, "eval_attn_topk_k", 5))
        )
        self.eval_attn_topk_row_stride = max(
            1, int(getattr(config, "eval_attn_topk_row_stride", 1))
        )
        self.eval_attn_topk_max_rows = max(
            0, int(getattr(config, "eval_attn_topk_max_rows", 0))
        )
        self.eval_attn_topk_include_base = self._as_bool(
            getattr(config, "eval_attn_topk_include_base", True), default=True
        )
        self.eval_attn_topk_decode_tokens = self._as_bool(
            getattr(config, "eval_attn_topk_decode_tokens", True), default=True
        )
        self.eval_attn_topk_token_max_chars = max(
            8, int(getattr(config, "eval_attn_topk_token_max_chars", 64))
        )
        self.eval_attn_topk_export_full_matrix = self._as_bool(
            getattr(config, "eval_attn_topk_export_full_matrix", False), default=False
        )
        self.eval_attn_topk_full_matrix_head_limit = max(
            1, int(getattr(config, "eval_attn_topk_full_matrix_head_limit", 1))
        )
        self.eval_attn_topk_export_qa_text = self._as_bool(
            getattr(config, "eval_attn_topk_export_qa_text", True), default=True
        )
        self.eval_attn_pattern_feature_enabled = self._as_bool(
            getattr(config, "eval_attn_pattern_feature_enabled", False), default=False
        )
        self.eval_attn_pattern_feature_query_stride = max(
            1, int(getattr(config, "eval_attn_pattern_feature_query_stride", 1))
        )
        self.eval_attn_pattern_debug_tensors = self._as_bool(
            getattr(config, "eval_attn_pattern_debug_tensors", False), default=False
        )
        self.eval_attn_pattern_debug_only_problem_layers = self._as_bool(
            getattr(config, "eval_attn_pattern_debug_only_problem_layers", True), default=True
        )
        self.eval_attn_pattern_dataset_jsonl = str(
            getattr(config, "eval_attn_pattern_dataset_jsonl", "")
        ).strip()
        self._eval_attn_tokenizer = None
        self._eval_attn_tokenizer_ready = False
        self._eval_attn_tokenizer_name = None
        self._eval_attn_pattern_dataset_cache = None
        self._eval_attn_heatmap_export_count = 0
        self.eval_attn_mech_enabled = self._as_bool(
            getattr(config, "eval_attn_mech_enabled", getattr(config, "eval_attn_heatmap_enabled", False)),
            default=False,
        )
        self.eval_attn_mech_eps = float(getattr(config, "eval_attn_mech_eps", 1e-12))
        self.wavelet_ctxscale_abs_shift_causal = self._as_bool(
            getattr(config, "wavelet_ctxscale_abs_shift_causal", False), default=False
        )
        self.wavelet_ctxscale_film_hidden = max(8, int(getattr(config, "wavelet_ctxscale_film_hidden", 64)))
        self.wavelet_ctxscale_film_alpha = float(getattr(config, "wavelet_ctxscale_film_alpha", 0.5))
        self.wavelet_ctxscale_film_beta = float(getattr(config, "wavelet_ctxscale_film_beta", 0.1))
        self.wavelet_ctxscale_film_clamp = float(getattr(config, "wavelet_ctxscale_film_clamp", 8.0))
        self.wavelet_ctxscale_far_only = self._as_bool(
            getattr(config, "wavelet_ctxscale_far_only", False), default=False
        )
        self.wavelet_ctxscale_far_min_delta = max(0, int(getattr(config, "wavelet_ctxscale_far_min_delta", 0)))
        # Eval-time controllable attenuation for very long distances:
        # if delta > wavelet_ctxscale_far_over_delta, multiply wavelet bias by wavelet_ctxscale_far_over_alpha.
        self.wavelet_ctxscale_far_over_delta = max(
            0, int(getattr(config, "wavelet_ctxscale_far_over_delta", 0))
        )
        self.wavelet_ctxscale_far_over_alpha = float(
            getattr(config, "wavelet_ctxscale_far_over_alpha", 1.0)
        )
        head_cfg = getattr(config, "wavelet_ctxscale_head_indices", "all")
        if isinstance(head_cfg, str) and head_cfg.strip().lower() in ("", "all", "*", "none"):
            self.wavelet_ctxscale_head_indices = None
        else:
            parsed_heads = self._parse_int_list(head_cfg, default=[])
            self.wavelet_ctxscale_head_indices = (
                sorted(set(int(h) for h in parsed_heads)) if len(parsed_heads) > 0 else None
            )
        self.wavelet_condfilm_v2_hidden = max(8, int(getattr(config, "wavelet_condfilm_v2_hidden", 128)))
        self.wavelet_condfilm_v2_alpha = float(getattr(config, "wavelet_condfilm_v2_alpha", 0.1))
        self.wavelet_condfilm_v2_beta = float(getattr(config, "wavelet_condfilm_v2_beta", 0.1))
        self.wavelet_condfilm_v2_clamp = float(getattr(config, "wavelet_condfilm_v2_clamp", 8.0))
        self.wavelet_condfilm_v2_per_token_scalar = self._as_bool(
            getattr(config, "wavelet_condfilm_v2_per_token_scalar", True), default=True
        )
        self.wavelet_condfilm_v2_print_every = max(1, int(getattr(config, "wavelet_condfilm_v2_print_every", 100)))
        self.wavelet_condfilm_v2_lock_window = max(1, int(getattr(config, "wavelet_condfilm_v2_lock_window", 200)))
        self.wavelet_condfilm_v2_lock_sat_thresh = float(
            getattr(config, "wavelet_condfilm_v2_lock_sat_thresh", 0.7)
        )
        self.wavelet_condfilm_v2_lock_grad_eps = float(
            getattr(config, "wavelet_condfilm_v2_lock_grad_eps", 1e-6)
        )
        self.wavelet_condfilm_v2_lr_mult = float(getattr(config, "wavelet_condfilm_v2_lr_mult", 0.1))
        self.wavelet_condfilm_v2_grad_clip_value = float(
            getattr(config, "wavelet_condfilm_v2_grad_clip_value", 0.0)
        )
        self.wavelet_condfilm_v2_use_full_backward_hook = self._as_bool(
            getattr(config, "wavelet_condfilm_v2_use_full_backward_hook", False), default=False
        )
        self.wavelet_ctxscale_lock_window = max(1, int(getattr(config, "wavelet_ctxscale_lock_window", 300)))
        self.wavelet_ctxscale_lock_grad_eps = float(getattr(config, "wavelet_ctxscale_lock_grad_eps", 1e-6))
        self.wavelet_ctxscale_lock_update_eps = float(getattr(config, "wavelet_ctxscale_lock_update_eps", 1e-6))
        self.wavelet_ctxscale_lock_min_frac = float(getattr(config, "wavelet_ctxscale_lock_min_frac", 0.5))
        self.wavelet_ctxscale_lock_clamp_abs = float(getattr(config, "wavelet_ctxscale_lock_clamp_abs", 8.0))
        self.wavelet_gate_grad_clip = float(getattr(config, "wavelet_gate_grad_clip", 1.0))
        self.wavelet_gate_autofix = self._as_bool(getattr(config, "wavelet_gate_autofix", False), default=False)
        self.wavelet_gate_autofix_clamp_abs = float(
            getattr(config, "wavelet_gate_autofix_clamp_abs", self.wavelet_ctxscale_lock_clamp_abs)
        )
        self._wavelet_delta_index_cache = {}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.r = int(num_harmonics)
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        logit_bias_a_init = float(getattr(config, "wavelet_logit_bias_a_init", -5.0))
        self.wavelet_logit_bias_a = nn.Parameter(torch.tensor(logit_bias_a_init, dtype=torch.float32))
        if self.wavelet_ctxscale_use_head_gate:
            self.wavelet_logit_bias_a_head = nn.Parameter(
                torch.full((self.num_heads,), logit_bias_a_init, dtype=torch.float32)
            )
        else:
            self.wavelet_logit_bias_a_head = None
        self._wavelet_gate_local_step = 0
        self._wavelet_gate_prev_a = None
        self._wavelet_gate_prev_step = None
        self._wavelet_gate_last_grad_abs = None
        self._wavelet_gate_last_grad_p50 = None
        self._wavelet_gate_last_grad_p90 = None
        self._wavelet_gate_last_grad_max = None
        self._wavelet_gate_last_grad_zero_ratio = None
        self._wavelet_gate_last_grad_finite_ratio = None
        self._wavelet_gate_last_grad_nonfinite = None
        self._wavelet_gate_grad_seen = False
        self._wavelet_gate_last_metrics = None
        self._wavelet_gate_last_metrics_step = None
        self._wavelet_gate_locked = False
        self._wavelet_gate_last_nf_warn_step = None
        self._wavelet_gate_last_missing_warn_step = None
        self._wavelet_gate_hist = deque(maxlen=self.wavelet_ctxscale_lock_window)
        self._wavelet_gate_grad_hook_handle = self.wavelet_logit_bias_a.register_hook(
            self._capture_wavelet_gate_grad
        )
        self._wavelet_gate_grad_hook_param_id = id(self.wavelet_logit_bias_a)
        # PAT-225: fixed-support log-uniform grid over [2^0, 2^14] for any K.
        # K=8 -> exponents 14*i/7 == 2*i, i.e. the production grid [2^0,2^2,...,2^14]
        # reproduced bit-exactly. K=1 -> geometric center of the support, 2^7 = 128
        # (pre-registered in PAT-225). Endpoints stay fixed for every K>1 so that
        # scale cardinality is the only changed factor.
        _K = self.wavelet_ctxscale_k
        self.multiscale_norm_requested = str(
            getattr(config, "multiscale_norm", "none")
        ).strip().lower()
        self.wavelet_ctxscale_amplitude_multiplier = float(
            getattr(
                config,
                "wavelet_ctxscale_amplitude_multiplier",
                getattr(config, "amplitude_multiplier", -1.0),
            )
        )
        if not math.isfinite(self.wavelet_ctxscale_amplitude_multiplier):
            raise ValueError(
                "wavelet_ctxscale_amplitude_multiplier/amplitude_multiplier "
                f"must be finite, got {self.wavelet_ctxscale_amplitude_multiplier}"
            )
        self.wavelet_ctxscale_amplitude_multiplier_override = (
            self.wavelet_ctxscale_amplitude_multiplier != -1.0
        )

        self.multiscale_sum_scale = self._get_multiscale_sum_scale(
            self.multiscale_norm_requested,
            _K,
        )
        # PAT-227: support upper bound is configurable (log2 exponent; default 14
        # keeps every existing grid bit-identical). K=1 -> geometric center.
        _max_exp = getattr(
            config,
            "wavelet_ctxscale_scale_max_exp",
            14.0,
        )

        if isinstance(_max_exp, (list, tuple)):
            if len(_max_exp) != _K:
                raise ValueError(
                    f"wavelet_ctxscale_scale_max_exp must have "
                    f"{_K} elements, not {len(_max_exp)}: {_max_exp}"
                )
            _scale_exps = [float(x) / 2.0 for x in _max_exp]
        else:
            if _K != 1:
                raise ValueError(
                    f"wavelet_ctxscale_scale_max_exp must be a list/tuple of length {_K} when wavelet_ctxscale_k={_K}, not a single value: {_max_exp}"
                )
            _max_exp = float(_max_exp)
            _scale_exps = [_max_exp / 2.0]

        # PAT-244: expand each distinct scale into shift_number consecutive
        # copies (repeat, not tile: [e0,e0,e0,e1,e1,e1] for K=2,S=3) -- the
        # forward-pass router-output expansion below uses repeat_interleave
        # with the same consecutive-block ordering, so this buffer's layout
        # must match it exactly for the pi<->scale correspondence to hold.
        _shift_number = self.wavelet_ctxscale_shift_number
        _scale_exps_expanded = [e for e in _scale_exps for _ in range(_shift_number)]

        # PAT-244: optionally make the scale exponent itself a learned parameter
        # instead of a fixed buffer. Parametrized in log2-exponent space (not
        # raw scale) so positivity is automatic (2**x > 0 for any real x) and
        # gradient steps stay consistent with how this session's whole
        # scale-sweep analysis operates (u_edge, half-life multipliers, etc.
        # are all reasoned about in exponent/log2 space).
        self.wavelet_ctxscale_learnable_scale = self._as_bool(
            getattr(config, "wavelet_ctxscale_learnable_scale", False), default=False
        )
        if self.wavelet_ctxscale_learnable_scale:
            self.wavelet_ctxscale_scale_exp = nn.Parameter(
                torch.tensor(_scale_exps_expanded, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "wavelet_ctxscale_scales",
                torch.tensor([2.0 ** e * SCALE_MULTIPLIER_DICT[self.bias_type] for e in _scale_exps_expanded], dtype=torch.float32),
                persistent=False,
            )
        if self.wavelet_ctxscale_fixed_scale_ratio is not None:
            if len(self.wavelet_ctxscale_fixed_scale_ratio) != _K:
                raise ValueError(
                    f"wavelet_ctxscale_fixed_scale_ratio must have {_K} elements, "
                    f"not {len(self.wavelet_ctxscale_fixed_scale_ratio)}: {self.wavelet_ctxscale_fixed_scale_ratio}"
                )
            _fixed_scale_ratio = torch.tensor(
                self.wavelet_ctxscale_fixed_scale_ratio, dtype=torch.float32
            )
            _fixed_scale_ratio_sum = float(_fixed_scale_ratio.sum().item())
            if (not math.isfinite(_fixed_scale_ratio_sum)) or _fixed_scale_ratio_sum <= 0.0:
                raise ValueError(
                    "wavelet_ctxscale_fixed_scale_ratio must sum to a positive finite value, "
                    f"got {_fixed_scale_ratio_sum} from {self.wavelet_ctxscale_fixed_scale_ratio}"
                )
            _fixed_scale_ratio = _fixed_scale_ratio / _fixed_scale_ratio_sum
            self.register_buffer(
                "wavelet_ctxscale_fixed_scale_ratio_buf",
                _fixed_scale_ratio,
                persistent=False,
            )
        else:
            self.wavelet_ctxscale_fixed_scale_ratio_buf = None
        if self.wavelet_ctxscale_ratio_learnable:
            # Raw logits (not the ratio itself), passed through sigmoid at use-site --
            # same non-negative-independent-per-scale parameterization as
            # with_null_independent_scales' g_l, just query-independent (init 0 ->
            # sigmoid(0)=0.5 each, unbiased starting point).
            self.wavelet_ctxscale_ratio_param = nn.Parameter(torch.zeros(_K))
        if self.wavelet_ctxscale_g0_learnable:
            # Raw logit (not the gate itself), sigmoid at use-site -- init 0 ->
            # sigmoid(0)=0.5, matching the scale-ratio param's unbiased starting point.
            self.wavelet_ctxscale_g0_param = nn.Parameter(torch.zeros(1))
        if layer_idx in (0, None):
            print(
                f"[PAT-225] wavelet_ctxscale_k={_K} shift_number={_shift_number} "
                f"learnable_scale={self.wavelet_ctxscale_learnable_scale} effective scales="
                f"{[float(2.0 ** e * SCALE_MULTIPLIER_DICT[self.bias_type]) for e in _scale_exps_expanded]}",
                flush=True,
            )
            if self.wavelet_ctxscale_fixed_scale_ratio_buf is not None:
                print(
                    f"[PAT-225] wavelet_ctxscale_fixed_scale_ratio="
                    f"{self.wavelet_ctxscale_fixed_scale_ratio_buf.detach().cpu().tolist()}",
                    flush=True,
                )
        self.wavelet_ctx_feat_ln = nn.LayerNorm(self.head_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.wavelet_ctx_path_ln = nn.LayerNorm(3 * self.head_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.wavelet_ctx_path_proj = nn.Linear(3 * self.head_dim, self.head_dim, bias=True)
        self.wavelet_ctx_router = nn.Linear(self.head_dim, self.wavelet_ctxscale_k + 1, bias=True)
        # PAT-225 follow-up: router defaults to nn.Linear's kaiming_uniform init
        # (bound +/-1/sqrt(head_dim), ~125x larger std than wavelet_bias_film's
        # explicit 1e-3 init). Opt-in zero init tests whether that larger init
        # variance is what makes the router's converged state seed/hardware-chaotic.
        if self._as_bool(getattr(config, "wavelet_ctx_router_zero_init", False), default=False):
            nn.init.zeros_(self.wavelet_ctx_router.weight)
            nn.init.zeros_(self.wavelet_ctx_router.bias)
        # E2b ablation: static globally-learned router (not query-conditioned)
        self.wavelet_router_static_learned = self._as_bool(
            getattr(config, "wavelet_router_static_learned", False), default=False
        )
        if self.wavelet_router_static_learned:
            self.wavelet_static_router_logits = nn.Parameter(
                torch.zeros(self.wavelet_ctxscale_k + 1, dtype=torch.float32)
            )
        else:
            self.wavelet_static_router_logits = None
        # PAT-225 mixed-length-training follow-up: give the router an explicit signal for
        # the current forward pass's sequence length T, so it has a chance to learn a
        # length-conditioned scale preference (e.g. "prefer scale i at short T, scale j at
        # long T") instead of only ever seeing query-content features that carry no direct
        # T information. Off by default; zero-initialized so enabling it is a true no-op
        # until training moves the weights (matches every other opt-in flag added this
        # session). Only meaningful when training mixes multiple sequence lengths.
        self.wavelet_router_length_aware = self._as_bool(
            getattr(config, "wavelet_router_length_aware", False), default=False
        )
        if self.wavelet_router_length_aware:
            self.wavelet_ctx_router_length = nn.Linear(1, self.wavelet_ctxscale_k + 1, bias=False)
            nn.init.zeros_(self.wavelet_ctx_router_length.weight)
        else:
            self.wavelet_ctx_router_length = None
        self.wavelet_shift_ln = nn.LayerNorm(self.hidden_size, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        # PAT-244: shift_proj needs one output per actual wavelet slot
        # (k_total = k_distinct * shift_number), not per distinct scale --
        # router sizing above stays k_distinct+1 since the router only makes
        # one decision per distinct scale.
        _shift_proj_out = self.wavelet_ctxscale_k_total if self.wavelet_ctxscale_shift_per_scale else 1
        self.wavelet_shift_proj = nn.Linear(self.hidden_size, _shift_proj_out, bias=True)
        film_in_dim = self.wavelet_ctxscale_k + 2
        self.wavelet_bias_film_ln = nn.LayerNorm(film_in_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.wavelet_bias_film = nn.Sequential(
            nn.Linear(film_in_dim, self.wavelet_ctxscale_film_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(self.wavelet_ctxscale_film_hidden, 2, bias=True),
        )
        nn.init.normal_(self.wavelet_bias_film[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.wavelet_bias_film[-1].bias)
        # Param-matched non-wavelet bias baseline branch.
        self.mlp_bias_ctx_feat_ln = nn.LayerNorm(self.head_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp_bias_ctx_path_ln = nn.LayerNorm(3 * self.head_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp_bias_ctx_path_proj = nn.Linear(3 * self.head_dim, self.head_dim, bias=True)
        self.mlp_bias_router = nn.Linear(self.head_dim, self.wavelet_ctxscale_k + 1, bias=True)
        self.mlp_bias_shift_ln = nn.LayerNorm(self.hidden_size, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp_bias_shift_proj = nn.Linear(self.hidden_size, 1, bias=True)
        self.mlp_bias_logit_bias_a = nn.Parameter(torch.tensor(logit_bias_a_init, dtype=torch.float32))
        if self.wavelet_ctxscale_use_head_gate:
            self.mlp_bias_logit_bias_a_head = nn.Parameter(
                torch.full((self.num_heads,), logit_bias_a_init, dtype=torch.float32)
            )
        else:
            self.mlp_bias_logit_bias_a_head = None
        self.mlp_bias_film_ln = nn.LayerNorm(film_in_dim, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp_bias_film = nn.Sequential(
            nn.Linear(film_in_dim, self.wavelet_ctxscale_film_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(self.wavelet_ctxscale_film_hidden, 2, bias=True),
        )
        nn.init.normal_(self.mlp_bias_film[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp_bias_film[-1].bias)
        self.mlp_bias_basis_mlp = nn.Sequential(
            nn.Linear(1, self.head_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.head_dim, self.wavelet_ctxscale_k, bias=True),
        )
        self.mlp_bias_basis_ln = nn.LayerNorm(self.wavelet_ctxscale_k, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self._mlp_bias_param_count_printed = False
        self._mlp_bias_param_target = self._ctxscale_param_count(
            use_mlp=False,
            include_film=bool(self.wavelet_mode_resolved == "logit_bias_ctxscale_shift_v0_film"),
        )
        self._mlp_bias_param_current = self._ctxscale_param_count(
            use_mlp=True,
            include_film=bool(self.wavelet_mode_resolved == "logit_bias_ctxscale_shift_v0_film"),
        )
        mlp_pad_size = max(0, int(self._mlp_bias_param_target - self._mlp_bias_param_current))
        self.mlp_bias_param_pad = nn.Parameter(torch.zeros((mlp_pad_size,), dtype=torch.float32))
        self.wavelet_cond_film_v2 = None
        self._wavelet_condfilm_v2_last_attn_stats = None
        if self.wavelet_mode_resolved == "cond_film_v2":
            self.wavelet_cond_film_v2 = WaveletCondFiLM_v2(
                d_model=self.hidden_size,
                d_wavelet=None,
                hidden=self.wavelet_condfilm_v2_hidden,
                alpha=self.wavelet_condfilm_v2_alpha,
                beta=self.wavelet_condfilm_v2_beta,
                clamp=self.wavelet_condfilm_v2_clamp,
                per_token_scalar=self.wavelet_condfilm_v2_per_token_scalar,
                print_every=self.wavelet_condfilm_v2_print_every,
                lock_window=self.wavelet_condfilm_v2_lock_window,
                lock_sat_thresh=self.wavelet_condfilm_v2_lock_sat_thresh,
                lock_grad_eps=self.wavelet_condfilm_v2_lock_grad_eps,
                grad_clip_value=self.wavelet_condfilm_v2_grad_clip_value,
                use_full_backward_hook=self.wavelet_condfilm_v2_use_full_backward_hook,
            )

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
                        if getattr(config, 'router_gate_use', False):
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
    @staticmethod
    def _resolve_multiscale_norm(
        multiscale_norm: str,
        router_sigmoid_mode: str,
        allow_with_null_multiscale_norm: bool = False,
    ) -> str:
        multiscale_norm = str(multiscale_norm).strip().lower()
        router_sigmoid_mode = str(router_sigmoid_mode).strip().lower()
        if multiscale_norm in ("sqrt_keff_detach", "keff_detach"):
            return multiscale_norm
        if (
            router_sigmoid_mode == "with_null"
            and not allow_with_null_multiscale_norm
        ):
            return "none"
        return multiscale_norm

    def _get_multiscale_sum_scale(self, multiscale_norm: str, K: int) -> float:
        if multiscale_norm in ("sqrt", "sqrt_k"):
            multiscale_sum_scale = 1.0 / math.sqrt(float(K))
        elif multiscale_norm == "k":
            multiscale_sum_scale = 1.0 / float(K)
        elif multiscale_norm in ("sqrt_keff_detach", "keff_detach"):
            # Runtime-dependent normalization is applied in forward.
            multiscale_sum_scale = 1.0
        elif multiscale_norm in ("rms", "rms_both"):
            # Runtime context-length RMS over the already summed multi-scale bias.
            multiscale_sum_scale = 1.0
        elif multiscale_norm == "gram":
            multiscale_sum_scale = float(
                self.wavelet_ctxscale_gram_alpha
            )
        else:
            multiscale_sum_scale = 1.0
        return multiscale_sum_scale

    @staticmethod
    def _get_sqrt_keff_detach_scale(
        g_chunk: torch.Tensor,
        *,
        eps: float,
        K: int,
    ) -> torch.Tensor:
        g_detached = g_chunk.detach().to(torch.float32)
        gate_sum = g_detached.sum(dim=-1, keepdim=True)
        gate_sq_sum = g_detached.square().sum(dim=-1, keepdim=True)
        k_eff = (
            gate_sum.square() / gate_sq_sum.clamp_min(eps)
        ).clamp(
            min=1.0,
            max=float(K),
        )
        return torch.rsqrt(k_eff)

    @staticmethod
    def _validate_dynamic_multiscale_norm_router(
        multiscale_norm: str,
        router_mode: str,
        *,
        intervention_active: bool = False,
    ) -> None:
        if multiscale_norm not in ("sqrt_keff_detach", "keff_detach"):
            return
        if router_mode != "sigmoid_with_null_independent_scales":
            raise ValueError(
                "sqrt_keff_detach is only supported for "
                "with_null_independent_scales routing"
            )
        if intervention_active:
            raise ValueError(
                "sqrt_keff_detach cannot be combined with scale intervention"
            )
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

    def _load_head_pair_csv(self, path):
        if path is None:
            return set()
        pairs = set()
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                pairs.add((int(row["layer"]), int(row["head"])))
        return pairs

    def _apply_eval_attn_norm(self, masked_logits, future, layer_idx):
        if self.attn_norm != "entmax":
            return torch.softmax(masked_logits, dim=-1)

        if self.entmax_scope == "all":
            return masked_entmax_bisect(masked_logits, alpha=self.entmax_alpha, valid_mask=~future, dim=-1)

        if self.entmax_scope == "deep_layers":
            if self.entmax_layers is None or int(layer_idx) in self.entmax_layers:
                return masked_entmax_bisect(masked_logits, alpha=self.entmax_alpha, valid_mask=~future, dim=-1)
            return torch.softmax(masked_logits, dim=-1)

        if self.entmax_scope == "stable_heads":
            p_softmax = torch.softmax(masked_logits, dim=-1)
            p_entmax = masked_entmax_bisect(masked_logits, alpha=self.entmax_alpha, valid_mask=~future, dim=-1)
            H = masked_logits.shape[1]
            is_stable = torch.tensor(
                [(int(layer_idx), h) in self.entmax_stable_heads for h in range(H)],
                device=masked_logits.device, dtype=torch.bool,
            )
            return torch.where(is_stable[None, :, None, None], p_entmax, p_softmax)

        if self.entmax_scope == "qwab_active_heads":
            raise NotImplementedError("PAT-195: qwab_active_heads not implemented")

        raise ValueError(f"unknown entmax_scope: {self.entmax_scope!r}")

    def _maybe_capture_entmax_attention_stats(self, probs, future, layer_idx):
        if not getattr(self, "_entmax_stats_capture", False):
            return

        probs_fp32 = probs.detach().to(torch.float32)
        valid_mask = ~future
        valid_rows = valid_mask.any(dim=-1)
        valid_counts = valid_mask.to(torch.float32).sum(dim=-1)
        safe_probs = torch.where(valid_mask, probs_fp32, torch.zeros_like(probs_fp32))
        safe_log_probs = torch.where(safe_probs > 0, safe_probs.clamp_min(torch.finfo(safe_probs.dtype).tiny).log(), torch.zeros_like(safe_probs))
        entropy = -(safe_probs * safe_log_probs).sum(dim=-1)
        log_valid_counts = torch.where(valid_counts > 1, valid_counts.log(), torch.ones_like(valid_counts))
        norm_entropy = torch.where(valid_counts > 1, entropy / log_valid_counts, torch.zeros_like(entropy))
        support_size = (safe_probs > 1e-6).to(torch.float32).sum(dim=-1)
        top32_mass = safe_probs.topk(k=min(32, safe_probs.shape[-1]), dim=-1).values.sum(dim=-1)
        top128_mass = safe_probs.topk(k=min(128, safe_probs.shape[-1]), dim=-1).values.sum(dim=-1)
        row_weight = valid_rows.to(torch.float32)
        n_rows = row_weight.sum(dim=(0, 2))
        denom = n_rows.clamp_min(1.0)
        self._entmax_last_stats = {
            "mean_entropy": (entropy * row_weight).sum(dim=(0, 2)).div(denom).cpu(),
            "mean_norm_entropy": (norm_entropy * row_weight).sum(dim=(0, 2)).div(denom).cpu(),
            "mean_support_size": (support_size * row_weight).sum(dim=(0, 2)).div(denom).cpu(),
            "mean_top32_mass": (top32_mass * row_weight).sum(dim=(0, 2)).div(denom).cpu(),
            "mean_top128_mass": (top128_mass * row_weight).sum(dim=(0, 2)).div(denom).cpu(),
            "n_rows": n_rows.cpu(),
        }

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
    def _normalize_path_attn_impl(impl) -> str:
        m = str(impl).strip().lower() if impl is not None else "pytorch"
        if m in ("triton", "cuda", "parallel", "parallel_path_attn", "fused"):
            return "triton"
        if m in ("pytorch", "torch", "python", "ref", "repro"):
            return "pytorch"
        return "pytorch"

    @staticmethod
    def _normalize_wavelet_mode(mode) -> str:
        m = str(mode).strip().lower() if mode is not None else "router_rel"
        if m in ("off", "none", "baseline"):
            return "off"
        if m in ("cond_film_v2", "film_v2", "wavelet_condfilm_v2", "waveletcondfilm_v2"):
            return "cond_film_v2"
        if m in (
            "logit_bias_ctxscale_shift_v0_film",
            "ctxscale_shift_v0_film",
            "ctxscale_shift_film",
            "ctxscale_shift_v0+film",
        ):
            return "logit_bias_ctxscale_shift_v0_film"
        if m in ("mlp_bias_baseline_v0", "mlp_bias_baseline", "ctxscale_mlp_bias_baseline_v0"):
            return "mlp_bias_baseline_v0"
        if m in ("logit_bias", "bias", "exp_a"):
            return "logit_bias"
        if m in ("logit_bias_ctxscale_shift_v0", "ctxscale_shift_v0", "ctxscale_shift"):
            return "logit_bias_ctxscale_shift_v0"
        # Keep legacy values routed to old relative-logit path.
        if m in ("router_rel", "router", "rel", "additive", "softmix"):
            return "router_rel"
        if m.startswith("db") or m.startswith("coif") or m.startswith("sym") or m == "haar":
            return "router_rel"
        return "router_rel"

    def _k1_emit_log(self, msg: str):
        # Avoid duplicated logs across DDP ranks.
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass
        logger_obj = getattr(self, "logger", None)
        if logger_obj is not None:
            try:
                logger_obj.info(msg)
                return
            except Exception:
                pass
        print(msg)

    def _wavelet_analysis_rank0(self) -> bool:
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return False
        except Exception:
            pass
        return True

    def _eval_attn_heatmap_rank0(self) -> bool:
        return self._wavelet_analysis_rank0()

    def _eval_attn_heatmap_layer_enabled(self, layer_idx: Optional[int]) -> bool:
        layers = getattr(self, "eval_attn_heatmap_layers", None)
        if layers is None:
            return True
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        return int(lid) in layers

    def _eval_attn_heatmap_emit(self, msg: str):
        logger_obj = getattr(self, "logger", None)
        if logger_obj is not None:
            try:
                logger_obj.info(msg)
                return
            except Exception:
                pass
        print(msg)

    @staticmethod
    def _infer_total_layers_from_config(cfg) -> Optional[int]:
        if cfg is None:
            return None
        for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            try:
                v = int(getattr(cfg, key))
                if v > 0:
                    return v
            except Exception:
                continue
        return None

    def _eval_attn_heatmap_block_tag(self) -> str:
        cfg = getattr(self, "config", None)
        block_size = None
        for key in ("block_size", "seq_len", "max_position_embeddings"):
            try:
                v = int(getattr(cfg, key))
                if v > 0:
                    block_size = v
                    break
            except Exception:
                continue
        if block_size is None:
            return "block_unknown"
        return f"block_{int(block_size)}"

    @staticmethod
    def _heatmap_quantile_or_max(
        x: torch.Tensor,
        *,
        quantile: float,
        min_value: float = 1e-8,
    ) -> float:
        xf = x.detach().reshape(-1)
        if xf.numel() <= 0:
            return float(min_value)
        xf = xf[torch.isfinite(xf)]
        if xf.numel() <= 0:
            return float(min_value)
        q = float(max(0.0, min(1.0, quantile)))
        if q >= 1.0:
            v = float(xf.max().item())
        else:
            v = float(torch.quantile(xf, q).item())
        if not math.isfinite(v):
            v = float(min_value)
        return float(max(v, min_value))

    @torch.no_grad()
    def _compute_output_delta_stats(
        self,
        *,
        out_base_bt_hd: torch.Tensor,
        out_wave_bt_hd: torch.Tensor,
    ) -> Optional[dict]:
        if out_base_bt_hd is None or out_wave_bt_hd is None:
            return None
        if out_base_bt_hd.dim() != 3 or out_wave_bt_hd.dim() != 3:
            return None
        if out_base_bt_hd.shape != out_wave_bt_hd.shape:
            return None

        eps = float(max(1e-12, float(getattr(self, "eval_attn_mech_eps", 1e-12))))
        ob = torch.nan_to_num(out_base_bt_hd.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        ow = torch.nan_to_num(out_wave_bt_hd.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        delta = ow - ob

        base_tok = torch.linalg.vector_norm(ob, ord=2, dim=-1)      # [T,H]
        delta_tok = torch.linalg.vector_norm(delta, ord=2, dim=-1)  # [T,H]
        rel_tok = delta_tok / base_tok.clamp_min(eps)
        rel_tok_head = rel_tok.mean(dim=0) if rel_tok.numel() > 0 else torch.zeros((ob.shape[1],), dtype=torch.float32)

        base_flat = ob.reshape(-1)
        wave_flat = ow.reshape(-1)
        delta_flat = delta.reshape(-1)
        base_l2 = torch.linalg.vector_norm(base_flat, ord=2)
        wave_l2 = torch.linalg.vector_norm(wave_flat, ord=2)
        delta_l2 = torch.linalg.vector_norm(delta_flat, ord=2)
        dot = torch.dot(base_flat, wave_flat)
        cos = dot / (base_l2 * wave_l2).clamp_min(eps)

        rel_q = _quantiles_flat(rel_tok.reshape(-1), qs=(0.5, 0.9, 0.99))
        return {
            "delta_o_l2_over_base_l2": float((delta_l2 / base_l2.clamp_min(eps)).item()),
            "delta_o_mean_token_l2": float(delta_tok.mean().item()),
            "base_o_mean_token_l2": float(base_tok.mean().item()),
            "delta_o_mean_token_l2_over_base": float((delta_tok.mean() / base_tok.mean().clamp_min(eps)).item()),
            "delta_o_rel_token_mean": float(rel_tok.mean().item()),
            "delta_o_rel_token_p50": float(rel_q["p50"]),
            "delta_o_rel_token_p90": float(rel_q["p90"]),
            "delta_o_rel_token_p99": float(rel_q["p99"]),
            "delta_o_cosine_flat": float(cos.item()),
            "delta_o_rel_token_head_mean": [float(x.item()) for x in rel_tok_head],
        }

    @torch.no_grad()
    def _eval_attn_get_tokenizer(self):
        if bool(getattr(self, "_eval_attn_tokenizer_ready", False)):
            return getattr(self, "_eval_attn_tokenizer", None)
        self._eval_attn_tokenizer_ready = True
        cfg = getattr(self, "config", None)
        cands = []
        for key in ("tokenizer_name", "model_name_or_path", "_name_or_path"):
            if cfg is None:
                continue
            val = str(getattr(cfg, key, "") or "").strip()
            if len(val) > 0 and val not in cands:
                cands.append(val)
        if "gpt2" not in cands:
            cands.append("gpt2")
        for name in cands:
            try:
                tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
                self._eval_attn_tokenizer = tok
                self._eval_attn_tokenizer_name = name
                return tok
            except Exception:
                continue
        self._eval_attn_tokenizer = None
        self._eval_attn_tokenizer_name = None
        return None

    @staticmethod
    def _eval_attn_clean_token_text(token_text: str, max_chars: int) -> str:
        s = str(token_text)
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        # GPT-2 BPE visible-space marker.
        s = s.replace("Ġ", "▁")
        if len(s) > int(max_chars):
            s = s[: int(max_chars) - 3] + "..."
        return s

    @torch.no_grad()
    def _eval_attn_extract_token_view(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        batch_index: int,
        need_len: int,
    ) -> tuple[list[int], list[str]]:
        need_len = max(0, int(need_len))
        if need_len <= 0:
            return [], []
        tok_ids = [-1 for _ in range(need_len)]
        tok_txt = ["" for _ in range(need_len)]
        if input_ids is None or (not torch.is_tensor(input_ids)) or input_ids.dim() != 2:
            for i in range(need_len):
                tok_txt[i] = f"pos:{i}"
            return tok_ids, tok_txt
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        if B <= 0 or T <= 0:
            for i in range(need_len):
                tok_txt[i] = f"pos:{i}"
            return tok_ids, tok_txt
        bidx = min(max(0, int(batch_index)), B - 1)
        take = min(int(need_len), int(T))
        ids_cpu = input_ids[bidx, :take].detach().to(device="cpu", dtype=torch.long).tolist()
        tok = self._eval_attn_get_tokenizer() if bool(getattr(self, "eval_attn_topk_decode_tokens", True)) else None
        max_chars = int(getattr(self, "eval_attn_topk_token_max_chars", 64))
        for i in range(take):
            tid = int(ids_cpu[i])
            tok_ids[i] = tid
            if tok is None:
                tok_txt[i] = f"id:{tid}"
                continue
            text = None
            try:
                text = tok.convert_ids_to_tokens(tid)
            except Exception:
                text = None
            if text is None:
                try:
                    text = tok.decode([tid], clean_up_tokenization_spaces=False)
                except Exception:
                    text = None
            if text is None:
                text = f"id:{tid}"
            tok_txt[i] = self._eval_attn_clean_token_text(text, max_chars=max_chars)
        for i in range(take, need_len):
            tok_txt[i] = f"pos:{i}"
        return tok_ids, tok_txt

    @torch.no_grad()
    def _eval_attn_extract_qa_text(self, decoded_text: str) -> dict:
        text = str(decoded_text or "")
        q_text = ""
        a_text = ""
        q_span = re.search(
            r"Question\s*:\s*(.*?)(?:\n+\s*Answer\s*:|Answer\s*:)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if q_span is not None:
            q_text = q_span.group(1).strip()
        a_span = re.search(
            r"Answer\s*:\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if a_span is not None:
            a_text = a_span.group(1).strip()
            # Trim common LM special token tails.
            a_text = re.split(r"<\|endoftext\|>|<\|eot_id\|>", a_text, maxsplit=1)[0].strip()
        return {
            "question": q_text,
            "answer": a_text,
        }

    @torch.no_grad()
    def _eval_attn_pattern_dataset_records(self) -> dict[int, dict]:
        cache = getattr(self, "_eval_attn_pattern_dataset_cache", None)
        if isinstance(cache, dict):
            return cache
        path = str(getattr(self, "eval_attn_pattern_dataset_jsonl", "") or "").strip()
        target_len = int(
            getattr(
                self,
                "eval_attn_pattern_target_len",
                getattr(self.config, "eval_attn_pattern_target_len", 0),
            ) or 0
        )
        records = {}
        if len(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as fin:
                    kept_idx = 0
                    for line in fin:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        rec_target = int(rec.get("meta", {}).get("target_total_tokens", -1))
                        if target_len > 0 and rec_target != target_len:
                            continue
                        records[int(kept_idx)] = rec
                        kept_idx += 1
            except Exception:
                records = {}
        self._eval_attn_pattern_dataset_cache = records
        return records

    @staticmethod
    def _pattern_support_mask(length: int, support_start: Optional[int], support_end: Optional[int], *, device) -> Optional[torch.Tensor]:
        if support_start is None or support_end is None:
            return None
        start = int(support_start)
        end = int(support_end)
        if start < 0 or end <= start or int(length) <= 0:
            return None
        mask = torch.zeros((int(length),), dtype=torch.bool, device=device)
        start = max(0, min(int(length), start))
        end = max(0, min(int(length), end))
        if end <= start:
            return None
        mask[start:end] = True
        return mask if bool(mask.any()) else None

    @staticmethod
    def _pattern_row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = torch.linalg.vector_norm(a, ord=2) * torch.linalg.vector_norm(b, ord=2)
        if float(denom.item()) <= 1e-8:
            return 0.0
        return float((a * b).sum().item() / denom.item())

    @staticmethod
    def _pattern_jsd(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.clamp_min(1e-8)
        b = b.clamp_min(1e-8)
        a = a / a.sum().clamp_min(1e-8)
        b = b / b.sum().clamp_min(1e-8)
        m = 0.5 * (a + b)
        kl_am = float((a * (a.log() - m.log())).sum().item())
        kl_bm = float((b * (b.log() - m.log())).sum().item())
        return 0.5 * (kl_am + kl_bm)

    @staticmethod
    def _pattern_adjacent_cos_mean(x: Optional[torch.Tensor], *, batch_index: int, t_take: int) -> Optional[float]:
        if x is None or (not torch.is_tensor(x)) or x.dim() != 4:
            return None
        if int(t_take) <= 1 or int(batch_index) >= int(x.shape[0]):
            return None
        xb = x[batch_index].detach().to(dtype=torch.float32, device="cpu")[: int(t_take)]
        if xb.dim() != 3 or int(xb.shape[0]) <= 1:
            return None
        x0 = xb[:-1]
        x1 = xb[1:]
        num = (x0 * x1).sum(dim=-1)
        den = x0.norm(dim=-1) * x1.norm(dim=-1)
        cos = torch.nan_to_num(num / den.clamp_min(1e-6), nan=0.0, posinf=0.0, neginf=0.0)
        if cos.numel() <= 0:
            return None
        return float(cos.mean().item())

    @torch.no_grad()
    def _compute_eval_attn_pattern_features(
        self,
        *,
        layer_agg: torch.Tensor,
        map_type: str,
        query_stride: int,
        support_start: Optional[int],
        support_end: Optional[int],
    ) -> dict:
        mat = torch.nan_to_num(layer_agg.detach().to(dtype=torch.float32, device="cpu"), nan=0.0, posinf=0.0, neginf=0.0)
        if mat.dim() != 2:
            raise ValueError(f"Expected [Q,K] matrix, got shape={tuple(mat.shape)}")
        q_len, k_len = int(mat.shape[-2]), int(mat.shape[-1])
        if q_len <= 0 or k_len <= 0:
            return {}
        rows = torch.zeros_like(mat)
        for i in range(q_len):
            vals = mat[i, : i + 1]
            if map_type == "raw_score":
                probs = torch.softmax(vals, dim=-1)
            else:
                probs = vals.clamp_min(0.0)
                probs = probs / probs.sum().clamp_min(1e-8)
            rows[i, : i + 1] = probs

        stride = max(1, int(query_stride))
        row_ids = list(range(0, q_len, stride))
        if (q_len - 1) not in row_ids:
            row_ids.append(q_len - 1)

        vertical_vals = []
        diagonal_vals = []
        jsd_vals = []
        support_vertical_vals = []
        nonsupport_vertical_vals = []
        aligned = torch.zeros((len(row_ids), k_len), dtype=torch.float32)
        support_mask = self._pattern_support_mask(k_len, support_start, support_end, device=rows.device)

        for ridx, q_idx in enumerate(row_ids):
            vals = rows[q_idx, : q_idx + 1]
            aligned[ridx, : q_idx + 1] = torch.flip(vals, dims=[0])
            if ridx == 0:
                continue
            prev_q = row_ids[ridx - 1]
            cur_q = q_idx
            prev_row = rows[prev_q]
            cur_row = rows[cur_q]
            v = self._pattern_row_cos(prev_row, cur_row)
            vertical_vals.append(v)
            shift = max(1, int(cur_q - prev_q))
            if shift < k_len:
                d = self._pattern_row_cos(prev_row[shift:], cur_row[:-shift])
            else:
                d = 0.0
            diagonal_vals.append(d)
            jsd_vals.append(self._pattern_jsd(prev_row, cur_row))
            if support_mask is not None:
                in_support = bool(support_mask[min(prev_q, support_mask.numel() - 1)].item()) or bool(
                    support_mask[min(cur_q, support_mask.numel() - 1)].item()
                )
                if in_support:
                    support_vertical_vals.append(v)
                else:
                    nonsupport_vertical_vals.append(v)

        if aligned.shape[0] > 2:
            spec = torch.fft.rfft(aligned, dim=0)
            mag = spec.abs()
            nonzero = mag[1:] if int(mag.shape[0]) > 1 else None
            col_mass = aligned.abs().sum(dim=0)
            keep = col_mass > 1e-8
            if nonzero is not None and bool(keep.any()):
                peak = nonzero[:, keep].max(dim=0).values
                dc = mag[0, keep].clamp_min(1e-8)
                periodic_seq = float((peak / dc).mean().item())
            else:
                periodic_seq = 0.0
            fft2 = torch.fft.rfft2(aligned)
            mag2 = fft2.abs()
            mag2[0, 0] = 0.0
            flat = mag2.flatten()
            total = float(flat.sum().item())
            if total > 1e-8:
                topk = min(8, int(flat.numel()))
                seasonal = float(torch.topk(flat, k=topk).values.sum().item() / total)
            else:
                seasonal = 0.0
        else:
            periodic_seq = 0.0
            seasonal = 0.0

        if support_mask is not None:
            support_key_mass = float(rows[:, support_mask].sum(dim=-1).mean().item())
        else:
            support_key_mass = float("nan")

        return {
            "vertical_score": float(sum(vertical_vals) / len(vertical_vals)) if vertical_vals else 0.0,
            "diagonal_score": float(sum(diagonal_vals) / len(diagonal_vals)) if diagonal_vals else 0.0,
            "unpredictability_score": float(1.0 - (sum(vertical_vals) / len(vertical_vals))) if vertical_vals else 1.0,
            "jsd_unpredictability": float(sum(jsd_vals) / len(jsd_vals)) if jsd_vals else 0.0,
            "periodic_seq_score": float(periodic_seq),
            "seasonal_2d_score": float(seasonal),
            "support_key_mass": float(support_key_mass),
            "support_vertical_score": float(sum(support_vertical_vals) / len(support_vertical_vals)) if support_vertical_vals else float("nan"),
            "nonsupport_vertical_score": float(sum(nonsupport_vertical_vals) / len(nonsupport_vertical_vals)) if nonsupport_vertical_vals else float("nan"),
            "query_stride": int(stride),
            "num_sampled_rows": int(len(row_ids)),
        }

    @torch.no_grad()
    def _export_eval_attn_pattern_features(
        self,
        *,
        out_root: Path,
        model_type: Optional[str],
        layer_idx: int,
        case_id: int,
        step_val: int,
        q_take: int,
        k_take: int,
        p_base_cpu: torch.Tensor,
        p_wav_cpu: torch.Tensor,
        zb_cpu: Optional[torch.Tensor],
        zw_cpu: Optional[torch.Tensor],
        q_repr: Optional[torch.Tensor],
        k_repr: Optional[torch.Tensor],
        batch_index: int,
        wavmass_mean: Optional[float],
        wavmass_p50: Optional[float],
        wavmass_p90: Optional[float],
        null_mass_mean: Optional[float],
        null_mass_p50: Optional[float],
        null_mass_p90: Optional[float],
        router_prob_finite_ratio: Optional[float],
        non_null_finite_ratio: Optional[float],
        null_finite_ratio: Optional[float],
        g_bias_finite_ratio: Optional[float],
        base_finite_ratio: Optional[float],
        router_stats_valid: Optional[bool],
    ) -> None:
        if not bool(getattr(self, "eval_attn_pattern_feature_enabled", False)):
            return
        query_stride = max(1, int(getattr(self, "eval_attn_pattern_feature_query_stride", 1)))
        dataset_records = self._eval_attn_pattern_dataset_records()
        ds_rec = dataset_records.get(int(case_id), {})
        ds_meta = ds_rec.get("meta", {}) if isinstance(ds_rec, dict) else {}
        support_start = ds_meta.get("support_start_tok")
        support_end = ds_meta.get("support_end_tok")
        q_adj = self._pattern_adjacent_cos_mean(q_repr, batch_index=int(batch_index), t_take=int(q_take))
        k_adj = self._pattern_adjacent_cos_mean(k_repr, batch_index=int(batch_index), t_take=int(k_take))
        model_type_norm = str(
            model_type
            or getattr(self, "eval_attn_pattern_feature_model_type", "")
            or getattr(self.config, "eval_attn_pattern_feature_model_type", "")
        ).strip().lower()
        if model_type_norm not in ("path_only", "path_wavelet"):
            raise ValueError(f"Unsupported PAT-82 model_type for export: {model_type_norm!r}")
        final_prob = p_base_cpu if model_type_norm == "path_only" else p_wav_cpu
        final_score = zb_cpu if model_type_norm == "path_only" else zw_cpu

        payload = {
            "meta": {
                "model_type": model_type_norm,
                "layer": int(layer_idx),
                "case_id": int(case_id),
                "step": int(step_val),
                "q_len": int(q_take),
                "k_len": int(k_take),
                "query_stride": int(query_stride),
                "support_start_tok": int(support_start) if support_start is not None else -1,
                "support_end_tok": int(support_end) if support_end is not None else -1,
                "example_id": str(ds_rec.get("_id", "")) if isinstance(ds_rec, dict) else "",
                "q_adj_cos_mean": float(q_adj) if q_adj is not None else float("nan"),
                "k_adj_cos_mean": float(k_adj) if k_adj is not None else float("nan"),
                "wavmass_mean": float(wavmass_mean) if wavmass_mean is not None else float("nan"),
                "wavmass_p50": float(wavmass_p50) if wavmass_p50 is not None else float("nan"),
                "wavmass_p90": float(wavmass_p90) if wavmass_p90 is not None else float("nan"),
                "non_null_mass_mean": float(wavmass_mean) if wavmass_mean is not None else float("nan"),
                "non_null_mass_p50": float(wavmass_p50) if wavmass_p50 is not None else float("nan"),
                "non_null_mass_p90": float(wavmass_p90) if wavmass_p90 is not None else float("nan"),
                "null_mass_mean": float(null_mass_mean) if null_mass_mean is not None else float("nan"),
                "null_mass_p50": float(null_mass_p50) if null_mass_p50 is not None else float("nan"),
                "null_mass_p90": float(null_mass_p90) if null_mass_p90 is not None else float("nan"),
                "router_prob_finite_ratio": float(router_prob_finite_ratio) if router_prob_finite_ratio is not None else float("nan"),
                "non_null_finite_ratio": float(non_null_finite_ratio) if non_null_finite_ratio is not None else float("nan"),
                "null_finite_ratio": float(null_finite_ratio) if null_finite_ratio is not None else float("nan"),
                "g_bias_finite_ratio": float(g_bias_finite_ratio) if g_bias_finite_ratio is not None else float("nan"),
                "base_finite_ratio": float(base_finite_ratio) if base_finite_ratio is not None else float("nan"),
                "router_stats_valid": bool(router_stats_valid) if router_stats_valid is not None else True,
            },
            "attention_prob": {
                "final": self._compute_eval_attn_pattern_features(
                    layer_agg=final_prob.mean(dim=0),
                    map_type="attention_prob",
                    query_stride=query_stride,
                    support_start=support_start,
                    support_end=support_end,
                ),
            },
        }
        if final_score is not None:
            payload["raw_score"] = {
                "final": self._compute_eval_attn_pattern_features(
                    layer_agg=final_score.mean(dim=0),
                    map_type="raw_score",
                    query_stride=query_stride,
                    support_start=support_start,
                    support_end=support_end,
                ),
            }
        with (out_root / "pattern_features.json").open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)
        if bool(getattr(self, "eval_attn_pattern_debug_tensors", False)):
            q_batch = None
            k_batch = None
            if torch.is_tensor(q_repr) and int(batch_index) < int(q_repr.shape[0]):
                q_batch = q_repr[batch_index]
            if torch.is_tensor(k_repr) and int(batch_index) < int(k_repr.shape[0]):
                k_batch = k_repr[batch_index]
            qk_stage_debug = getattr(self, "_last_pat82_qk_stage_debug", None)
            debug_payload = {
                "meta": {
                    "model_type": model_type_norm,
                    "layer": int(layer_idx),
                    "case_id": int(case_id),
                    "step": int(step_val),
                    "example_id": str(ds_rec.get("_id", "")) if isinstance(ds_rec, dict) else "",
                },
                "final_prob": _tensor_debug_summary_json("final_prob", final_prob),
                "final_score": _tensor_debug_summary_json("final_score", final_score),
                "p_base_cpu": _tensor_debug_summary_json("p_base_cpu", p_base_cpu),
                "p_wav_cpu": _tensor_debug_summary_json("p_wav_cpu", p_wav_cpu),
                "zb_cpu": _tensor_debug_summary_json("zb_cpu", zb_cpu),
                "zw_cpu": _tensor_debug_summary_json("zw_cpu", zw_cpu),
                "q_batch": _tensor_debug_summary_json("q_batch", q_batch),
                "k_batch": _tensor_debug_summary_json("k_batch", k_batch),
                "qk_stage_debug": qk_stage_debug,
            }
            def _iter_debug_nodes(node):
                if isinstance(node, dict):
                    if "finite_ratio" in node and isinstance(node.get("finite_ratio"), (int, float)):
                        yield node
                    for v in node.values():
                        yield from _iter_debug_nodes(v)
            problem = False
            for node in _iter_debug_nodes(debug_payload):
                fr = float(node.get("finite_ratio", float("nan")))
                if (not math.isfinite(fr)) or fr < 0.999999:
                    problem = True
                    break
            if (not bool(getattr(self, "eval_attn_pattern_debug_only_problem_layers", True))) or problem:
                with (out_root / "tensor_debug.json").open("w", encoding="utf-8") as fout:
                    json.dump(debug_payload, fout, ensure_ascii=False, indent=2)
                if problem:
                    layer_repr = int(layer_idx)
                    print(
                        f"[PAT82-debug] layer={layer_repr} case={int(case_id)} "
                        f"problematic debug tensors detected; wrote tensor_debug.json"
                    )

    @torch.no_grad()
    def _export_eval_attn_topk_table(
        self,
        *,
        out_root: Path,
        layer_idx: int,
        case_id: int,
        step_val: int,
        head_limit: int,
        q_take: int,
        k_take: int,
        p_base_cpu: torch.Tensor,   # [H,Q,K]
        p_wav_cpu: torch.Tensor,    # [H,Q,K]
        logit_delta_cpu: Optional[torch.Tensor] = None,  # [H,Q,K]
        input_ids: Optional[torch.Tensor] = None,        # [B,T]
        batch_index: int = 0,
    ):
        if int(head_limit) <= 0 or int(q_take) <= 0 or int(k_take) <= 0:
            return
        topk_k = max(1, int(getattr(self, "eval_attn_topk_k", 5)))
        row_stride = max(1, int(getattr(self, "eval_attn_topk_row_stride", 1)))
        max_rows = max(0, int(getattr(self, "eval_attn_topk_max_rows", 0)))
        include_base = bool(getattr(self, "eval_attn_topk_include_base", True))
        sources = ["wavelet", "base"] if include_base else ["wavelet"]

        need_len = max(int(q_take), int(k_take))
        tok_ids, tok_txt = self._eval_attn_extract_token_view(
            input_ids=input_ids,
            batch_index=int(batch_index),
            need_len=int(need_len),
        )
        tok = self._eval_attn_get_tokenizer() if bool(getattr(self, "eval_attn_topk_decode_tokens", True)) else None
        decoded_text = ""
        if tok is not None:
            valid_ids = [int(x) for x in tok_ids if int(x) >= 0]
            if len(valid_ids) > 0:
                try:
                    decoded_text = tok.decode(valid_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except Exception:
                    decoded_text = ""
        if bool(getattr(self, "eval_attn_topk_export_qa_text", True)):
            qa_payload = self._eval_attn_extract_qa_text(decoded_text=decoded_text)
            with (out_root / "qa_content.json").open("w", encoding="utf-8") as fout:
                json.dump(
                    {
                        "layer": int(layer_idx),
                        "case_id": int(case_id),
                        "step": int(step_val),
                        "question": str(qa_payload.get("question", "")),
                        "answer": str(qa_payload.get("answer", "")),
                        "decoded_text": str(decoded_text),
                    },
                    fout,
                    ensure_ascii=False,
                    indent=2,
                )

        out_csv = out_root / "topk_tokens.csv"
        out_rowmax = out_root / "rowmax_tokens.csv"
        out_freq = out_root / "top1_token_freq.csv"
        out_sum = out_root / "topk_summary_by_head.csv"
        out_tokens = out_root / "token_table.csv"

        top1_dist = {}
        top1_tok_counter = {}
        rowmax_records = []

        with out_tokens.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["pos", "token_id", "token"])
            for pos in range(int(need_len)):
                tid = int(tok_ids[pos]) if pos < len(tok_ids) else -1
                ttxt = tok_txt[pos] if pos < len(tok_txt) else f"pos:{pos}"
                writer.writerow([int(pos), int(tid), str(ttxt)])

        with out_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "layer",
                    "case_id",
                    "step",
                    "head",
                    "score_source",
                    "q_pos",
                    "q_token_id",
                    "q_token",
                    "rank",
                    "k_pos",
                    "k_token_id",
                    "k_token",
                    "key_distance",
                    "score_value",
                    "p_base",
                    "p_wavelet",
                    "p_delta",
                    "logit_delta",
                ]
            )
            for hid in range(int(head_limit)):
                pb = p_base_cpu[hid]
                pw = p_wav_cpu[hid]
                row_seen = 0
                for qpos in range(0, int(q_take), int(row_stride)):
                    if int(max_rows) > 0 and row_seen >= int(max_rows):
                        break
                    row_seen += 1
                    valid_k = min(int(k_take), int(qpos) + 1)  # causal valid range
                    if valid_k <= 0:
                        continue
                    q_tid = int(tok_ids[qpos]) if qpos < len(tok_ids) else -1
                    q_tok = tok_txt[qpos] if qpos < len(tok_txt) else f"pos:{qpos}"
                    for source in sources:
                        ref = pw if source == "wavelet" else pb
                        vec = ref[qpos, :valid_k]
                        cur_k = min(int(topk_k), int(valid_k))
                        vals, idxs = torch.topk(vec, k=cur_k, largest=True, sorted=True)
                        for ridx in range(int(cur_k)):
                            kpos = int(idxs[ridx].item())
                            score = float(vals[ridx].item())
                            p_b = float(pb[qpos, kpos].item())
                            p_w = float(pw[qpos, kpos].item())
                            p_d = float(p_w - p_b)
                            l_d = (
                                float(logit_delta_cpu[hid, qpos, kpos].item())
                                if logit_delta_cpu is not None
                                else 0.0
                            )
                            k_tid = int(tok_ids[kpos]) if kpos < len(tok_ids) else -1
                            k_tok = tok_txt[kpos] if kpos < len(tok_txt) else f"pos:{kpos}"
                            writer.writerow(
                                [
                                    int(layer_idx),
                                    int(case_id),
                                    int(step_val),
                                    int(hid),
                                    str(source),
                                    int(qpos),
                                    int(q_tid),
                                    str(q_tok),
                                    int(ridx + 1),
                                    int(kpos),
                                    int(k_tid),
                                    str(k_tok),
                                    int(qpos - kpos),
                                    score,
                                    p_b,
                                    p_w,
                                    p_d,
                                    l_d,
                                ]
                            )
                        if int(cur_k) > 0:
                            k0 = int(idxs[0].item())
                            key = (int(hid), str(source))
                            if key not in top1_dist:
                                top1_dist[key] = []
                                top1_tok_counter[key] = Counter()
                            top1_dist[key].append(int(qpos - k0))
                            tok0 = tok_txt[k0] if k0 < len(tok_txt) else f"pos:{k0}"
                            top1_tok_counter[key][str(tok0)] += 1
                            rowmax_records.append(
                                {
                                    "layer": int(layer_idx),
                                    "case_id": int(case_id),
                                    "step": int(step_val),
                                    "head": int(hid),
                                    "score_source": str(source),
                                    "q_pos": int(qpos),
                                    "q_token_id": int(q_tid),
                                    "q_token": str(q_tok),
                                    "max_k_pos": int(k0),
                                    "max_k_token_id": int(tok_ids[k0]) if k0 < len(tok_ids) else -1,
                                    "max_k_token": str(tok0),
                                    "key_distance": int(qpos - k0),
                                    "max_score": float(vec[k0].item()),
                                    "p_base": float(pb[qpos, k0].item()),
                                    "p_wavelet": float(pw[qpos, k0].item()),
                                    "p_delta": float((pw[qpos, k0] - pb[qpos, k0]).item()),
                                    "logit_delta": (
                                        float(logit_delta_cpu[hid, qpos, k0].item())
                                        if logit_delta_cpu is not None
                                        else 0.0
                                    ),
                                }
                            )

        with out_rowmax.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "layer",
                    "case_id",
                    "step",
                    "head",
                    "score_source",
                    "q_pos",
                    "q_token_id",
                    "q_token",
                    "max_k_pos",
                    "max_k_token_id",
                    "max_k_token",
                    "key_distance",
                    "max_score",
                    "p_base",
                    "p_wavelet",
                    "p_delta",
                    "logit_delta",
                ]
            )
            for rec in rowmax_records:
                writer.writerow(
                    [
                        rec["layer"],
                        rec["case_id"],
                        rec["step"],
                        rec["head"],
                        rec["score_source"],
                        rec["q_pos"],
                        rec["q_token_id"],
                        rec["q_token"],
                        rec["max_k_pos"],
                        rec["max_k_token_id"],
                        rec["max_k_token"],
                        rec["key_distance"],
                        rec["max_score"],
                        rec["p_base"],
                        rec["p_wavelet"],
                        rec["p_delta"],
                        rec["logit_delta"],
                    ]
                )

        if bool(getattr(self, "eval_attn_topk_export_full_matrix", False)):
            mat_head_limit = min(
                int(head_limit),
                max(1, int(getattr(self, "eval_attn_topk_full_matrix_head_limit", 1))),
            )
            for hid in range(int(mat_head_limit)):
                pb = p_base_cpu[hid]
                pw = p_wav_cpu[hid]
                for source in sources:
                    ref = pw if source == "wavelet" else pb
                    out_mat = out_root / f"head{hid:02d}_{source}_matrix.csv"
                    with out_mat.open("w", encoding="utf-8", newline="") as fout:
                        writer = csv.writer(fout)
                        header = ["q_pos", "q_token_id", "q_token"]
                        for kpos in range(int(k_take)):
                            ktid = int(tok_ids[kpos]) if kpos < len(tok_ids) else -1
                            ktok = tok_txt[kpos] if kpos < len(tok_txt) else f"pos:{kpos}"
                            header.append(f"k{int(kpos)}|id:{int(ktid)}|{ktok}")
                        writer.writerow(header)
                        for qpos in range(int(q_take)):
                            qtid = int(tok_ids[qpos]) if qpos < len(tok_ids) else -1
                            qtok = tok_txt[qpos] if qpos < len(tok_txt) else f"pos:{qpos}"
                            row = [int(qpos), int(qtid), str(qtok)]
                            vals = ref[qpos, :int(k_take)].tolist()
                            row.extend([float(v) for v in vals])
                            writer.writerow(row)

        with out_sum.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "layer",
                    "case_id",
                    "step",
                    "head",
                    "score_source",
                    "rows",
                    "top1_dist_mean",
                    "top1_dist_p50",
                    "top1_dist_p90",
                    "top1_dist_p99",
                    "top1_self_ratio",
                ]
            )
            for key in sorted(top1_dist.keys()):
                hid, source = key
                arr = torch.tensor(top1_dist[key], dtype=torch.float32)
                if arr.numel() <= 0:
                    continue
                p50 = float(torch.quantile(arr, 0.5).item())
                p90 = float(torch.quantile(arr, 0.9).item())
                p99 = float(torch.quantile(arr, 0.99).item())
                self_ratio = float((arr == 0).to(torch.float32).mean().item())
                writer.writerow(
                    [
                        int(layer_idx),
                        int(case_id),
                        int(step_val),
                        int(hid),
                        str(source),
                        int(arr.numel()),
                        float(arr.mean().item()),
                        p50,
                        p90,
                        p99,
                        self_ratio,
                    ]
                )

        with out_freq.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "layer",
                    "case_id",
                    "step",
                    "head",
                    "score_source",
                    "rank",
                    "token",
                    "count",
                    "ratio",
                ]
            )
            for key in sorted(top1_tok_counter.keys()):
                hid, source = key
                counter = top1_tok_counter[key]
                total = max(1, int(sum(counter.values())))
                for ridx, (tok, cnt) in enumerate(counter.most_common(50), start=1):
                    writer.writerow(
                        [
                            int(layer_idx),
                            int(case_id),
                            int(step_val),
                            int(hid),
                            str(source),
                            int(ridx),
                            str(tok),
                            int(cnt),
                            float(cnt / total),
                        ]
                    )

    @torch.no_grad()
    def _export_eval_attn_heatmaps(
        self,
        *,
        layer_idx: Optional[int],
        p_base: Optional[torch.Tensor],  # [B,H,T,T], softmaxed
        p_wav: Optional[torch.Tensor],   # [B,H,T,T], softmaxed
        logits_base: Optional[torch.Tensor] = None,  # [B,H,T,T], masked logits before softmax
        logits_wav: Optional[torch.Tensor] = None,   # [B,H,T,T], masked logits before softmax
        q_repr: Optional[torch.Tensor] = None,       # [B,T,H,D] or compatible
        k_repr: Optional[torch.Tensor] = None,       # [B,T,H,D] or compatible
        global_step=None,
        wavelet_mode: Optional[str] = None,
        has_wavelet: bool = False,
        rel_applied: bool = False,
        out_base: Optional[torch.Tensor] = None,  # [B,T,H,d]
        out_wav: Optional[torch.Tensor] = None,   # [B,T,H,d]
        input_ids: Optional[torch.Tensor] = None, # [B,T]
    ):
        heatmap_enabled = bool(getattr(self, "eval_attn_heatmap_enabled", False)) or bool(getattr(self, "_debug_enabled", False))
        if not heatmap_enabled:
            return
        if self.training:
            return
        if not self._eval_attn_heatmap_rank0():
            return
        if p_base is None or p_wav is None:
            return
        if p_base.dim() != 4 or p_wav.dim() != 4:
            return
        if int(getattr(self, "_eval_attn_heatmap_export_count", 0)) >= int(self.eval_attn_heatmap_case_limit):
            return
        if not self._eval_attn_heatmap_layer_enabled(layer_idx):
            return

        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        B, H, Tq, Tk = p_wav.shape
        if B <= 0 or H <= 0 or Tq <= 0 or Tk <= 0:
            return
        bidx = min(int(self.eval_attn_heatmap_case_index), B - 1)
        causal_mask_2d = torch.triu(
            torch.ones((Tq, Tk), device=p_wav.device, dtype=torch.bool), diagonal=1
        )

        def _safe_causal_logits_for_export(x: torch.Tensor) -> torch.Tensor:
            x = x.detach().to(dtype=torch.float32, device="cpu")
            q_len, k_len = int(x.shape[-2]), int(x.shape[-1])
            mask = causal_mask_2d[:q_len, :k_len].to(device=x.device)
            lower = ~mask
            fill = causal_mask_fill_value(x.dtype)
            finite_lower = torch.isfinite(x) & lower.unsqueeze(0)
            safe = torch.full_like(x, fill)
            safe = torch.where(finite_lower, x, safe)
            # Guarantee at least one valid key per query row for export-only softmax.
            diag_len = min(q_len, k_len)
            if diag_len > 0:
                d = torch.arange(diag_len, device=x.device)
                safe[:, d, d] = torch.where(
                    torch.isfinite(x[:, d, d]),
                    x[:, d, d],
                    torch.zeros_like(x[:, d, d]),
                )
            return safe.contiguous()

        def _safe_causal_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
            logits = _safe_causal_logits_for_export(logits)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            return probs.contiguous()

        def _masked_reduce_for_pt_export(x: Optional[torch.Tensor], *, is_logit: bool) -> Optional[torch.Tensor]:
            if x is None:
                return None
            x = x[:head_limit]
            if is_logit:
                x = _safe_causal_logits_for_export(x)
            else:
                x = torch.nan_to_num(
                    x.detach().to(dtype=torch.float32, device="cpu"),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            if not save_reduced_only:
                return x.contiguous()
            if x.dim() != 3:
                return x.contiguous()
            x2 = x.mean(dim=0, keepdim=False)
            if pt_resize_to > 0:
                q_len, k_len = int(x2.shape[-2]), int(x2.shape[-1])
                mask = (~causal_mask_2d[:q_len, :k_len]).to(device=x2.device, dtype=torch.float32)
                # Resize numerator and validity mask separately so masked upper-triangle
                # entries do not leak into the lower-triangular analysis map.
                x_num = F.interpolate(
                    (x2 * mask).unsqueeze(0).unsqueeze(0),
                    size=(pt_resize_to, pt_resize_to),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                x_den = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(pt_resize_to, pt_resize_to),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                x2 = x_num / x_den.clamp_min(1e-6)
                x2 = x2 * (x_den > 1e-6).to(dtype=x2.dtype)
            x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
            return x2.contiguous()

        def _adjacent_cosine_stats(x: Optional[torch.Tensor], *, t_take: int) -> Optional[dict]:
            if x is None or (not torch.is_tensor(x)) or x.dim() != 4:
                return None
            if int(t_take) <= 1 or int(bidx) >= int(x.shape[0]):
                return None
            xb = x[bidx].detach().to(dtype=torch.float32, device="cpu")
            xb = xb[: int(t_take)]
            if xb.dim() != 3 or xb.shape[0] <= 1:
                return None
            x0 = xb[:-1]
            x1 = xb[1:]
            num = (x0 * x1).sum(dim=-1)
            den = x0.norm(dim=-1) * x1.norm(dim=-1)
            cos = torch.nan_to_num(num / den.clamp_min(1e-6), nan=0.0, posinf=0.0, neginf=0.0)
            if cos.numel() <= 0:
                return None
            return {
                "mean": float(cos.mean().item()),
                "per_head_mean": [float(v.item()) for v in cos.mean(dim=0)],
            }

        p_base_cpu = p_base[bidx].detach().to(dtype=torch.float32, device="cpu")
        p_wav_cpu = p_wav[bidx].detach().to(dtype=torch.float32, device="cpu")
        q_take = min(int(p_base_cpu.shape[-2]), int(p_wav_cpu.shape[-2]))
        k_take = min(int(p_base_cpu.shape[-1]), int(p_wav_cpu.shape[-1]))
        if self.eval_attn_heatmap_max_seq > 0:
            q_take = min(q_take, int(self.eval_attn_heatmap_max_seq))
            k_take = min(k_take, int(self.eval_attn_heatmap_max_seq))
        p_base_cpu = p_base_cpu[:, :q_take, :k_take]
        p_wav_cpu = p_wav_cpu[:, :q_take, :k_take]
        logit_delta_cpu = None
        zb_cpu = None
        zw_cpu = None
        if logits_base is not None and logits_wav is not None and logits_base.dim() == 4 and logits_wav.dim() == 4:
            zb_cpu = logits_base[bidx].detach().to(dtype=torch.float32, device="cpu")[:, :q_take, :k_take]
            zw_cpu = logits_wav[bidx].detach().to(dtype=torch.float32, device="cpu")[:, :q_take, :k_take]
            # Export-only: rebuild probability maps from sanitized causal logits so
            # analysis payloads remain usable even when masked interpolation or
            # invalid rows would otherwise collapse them to NaN.
            p_base_cpu = _safe_causal_probs_from_logits(zb_cpu)
            p_wav_cpu = _safe_causal_probs_from_logits(zw_cpu)
            logit_delta_cpu = torch.nan_to_num(zw_cpu - zb_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            p_base_cpu = torch.nan_to_num(p_base_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            p_wav_cpu = torch.nan_to_num(p_wav_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        p_delta_cpu = p_wav_cpu - p_base_cpu
        output_delta_stats = None
        ob_cpu = None
        ow_cpu = None
        od_cpu = None
        wavmass_mean = None
        wavmass_p50 = None
        wavmass_p90 = None
        if out_base is not None and out_wav is not None and out_base.dim() == 4 and out_wav.dim() == 4:
            if out_base.shape == out_wav.shape:
                t_take_o = min(int(out_base.shape[1]), q_take)
                h_take_o = min(int(out_base.shape[2]), int(out_wav.shape[2]))
                ob_cpu = out_base[bidx].detach().to(dtype=torch.float32, device="cpu")[:t_take_o, :h_take_o, :]
                ow_cpu = out_wav[bidx].detach().to(dtype=torch.float32, device="cpu")[:t_take_o, :h_take_o, :]
                od_cpu = torch.nan_to_num(ow_cpu - ob_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                output_delta_stats = self._compute_output_delta_stats(
                    out_base_bt_hd=ob_cpu,
                    out_wave_bt_hd=ow_cpu,
                )

        head_limit = H if int(self.eval_attn_heatmap_head_limit) <= 0 else min(H, int(self.eval_attn_heatmap_head_limit))
        case_id = int(self._eval_attn_heatmap_export_count)
        step_val = self._to_int_or_none(global_step)
        if step_val is None:
            step_val = -1
        mode_name = str(wavelet_mode if wavelet_mode is not None else getattr(self, "wavelet_mode_resolved", "unknown"))
        block_tag = self._eval_attn_heatmap_block_tag()
        out_root = (
            Path(str(self.eval_attn_heatmap_outdir))
            / block_tag
            / str(self.eval_attn_heatmap_run_tag)
        )
        if bool(getattr(self, "eval_attn_heatmap_separate_step", False)):
            out_root = out_root / f"step{int(step_val):07d}"
        out_root = out_root / f"layer{lid:02d}" / f"case{case_id:03d}"

        try:
            out_root.mkdir(parents=True, exist_ok=True)
            delta_abs = p_delta_cpu[:head_limit].abs()
            delta_head_l1 = delta_abs.mean(dim=(-2, -1))
            logit_delta_abs = None
            logit_delta_head_l1 = None
            if logit_delta_cpu is not None:
                logit_delta_abs = logit_delta_cpu[:head_limit].abs()
                logit_delta_head_l1 = logit_delta_abs.mean(dim=(-2, -1))
            meta = {
                "layer": int(lid),
                "case_id": int(case_id),
                "step": int(step_val),
                "batch_index": int(bidx),
                "num_heads_total": int(H),
                "num_heads_saved": int(head_limit),
                "q_len": int(q_take),
                "k_len": int(k_take),
                "block_tag": str(block_tag),
                "wavelet_mode": mode_name,
                "has_wavelet": bool(has_wavelet),
                "rel_applied": bool(rel_applied),
                "delta_l1_mean": float(delta_abs.mean().item()) if int(head_limit) > 0 else 0.0,
                "delta_linf": float(delta_abs.max().item()) if int(head_limit) > 0 else 0.0,
                "delta_head_l1_mean": [float(x.item()) for x in delta_head_l1],
                "has_logit_delta": bool(logit_delta_cpu is not None),
                "has_output_delta": bool(output_delta_stats is not None),
                "topk_enabled": bool(getattr(self, "eval_attn_topk_enabled", True)),
                "topk_k": int(getattr(self, "eval_attn_topk_k", 5)),
                "topk_row_stride": int(getattr(self, "eval_attn_topk_row_stride", 1)),
                "topk_max_rows": int(getattr(self, "eval_attn_topk_max_rows", 0)),
                "topk_export_full_matrix": bool(getattr(self, "eval_attn_topk_export_full_matrix", False)),
                "topk_full_matrix_head_limit": int(getattr(self, "eval_attn_topk_full_matrix_head_limit", 1)),
                "topk_export_qa_text": bool(getattr(self, "eval_attn_topk_export_qa_text", True)),
            }
            do_val = getattr(self, "_last_ctxscale_do_validation", None)
            if isinstance(do_val, dict) and int(do_val.get("layer", -1)) == int(lid):
                meta["ctxscale_do_validation"] = do_val
            do_stat = getattr(self, "_last_ctxscale_do_stat", None)
            if isinstance(do_stat, dict) and int(do_stat.get("layer", -1)) == int(lid):
                meta["ctxscale_do_stat"] = do_stat
            q_adj = _adjacent_cosine_stats(q_repr, t_take=int(q_take))
            if isinstance(q_adj, dict):
                meta["q_adj_cos_mean"] = float(q_adj["mean"])
                meta["q_adj_cos_per_head_mean"] = list(q_adj["per_head_mean"])
            k_adj = _adjacent_cosine_stats(k_repr, t_take=int(k_take))
            if isinstance(k_adj, dict):
                meta["k_adj_cos_mean"] = float(k_adj["mean"])
                meta["k_adj_cos_per_head_mean"] = list(k_adj["per_head_mean"])
            ctx_prob = getattr(self, "_last_ctxscale_router_prob", None)
            ctx_non_null = getattr(self, "_last_ctxscale_non_null_mass", None)
            ctx_null = getattr(self, "_last_ctxscale_null_mass", None)
            ctx_payload = getattr(self, "_last_ctxscale_monitor_payload", None)
            router_prob_finite_ratio = float("nan")
            non_null_finite_ratio = float("nan")
            null_finite_ratio = float("nan")
            g_bias_finite_ratio = float("nan")
            base_finite_ratio = float("nan")
            router_stats_valid = True
            null_mass_mean = None
            null_mass_p50 = None
            null_mass_p90 = None
            if torch.is_tensor(ctx_prob):
                prob_all = ctx_prob[bidx].detach().to(dtype=torch.float32, device="cpu")
                router_prob_finite_ratio = float(torch.isfinite(prob_all).float().mean().item())
            if torch.is_tensor(ctx_non_null):
                mass = ctx_non_null[bidx].detach().to(dtype=torch.float32, device="cpu")
                non_null_finite_ratio = float(torch.isfinite(mass).float().mean().item())
                mass = torch.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
                wavmass_mean = float(mass.mean().item())
                wavmass_p50 = float(mass.quantile(0.5).item())
                wavmass_p90 = float(mass.quantile(0.9).item())
                meta["wavmass_mean"] = wavmass_mean
                meta["wavmass_p50"] = wavmass_p50
                meta["wavmass_p90"] = wavmass_p90
                meta["non_null_mass_mean"] = wavmass_mean
                meta["non_null_mass_p50"] = wavmass_p50
                meta["non_null_mass_p90"] = wavmass_p90
            if torch.is_tensor(ctx_null):
                nmass = ctx_null[bidx].detach().to(dtype=torch.float32, device="cpu")
                null_finite_ratio = float(torch.isfinite(nmass).float().mean().item())
                nmass = torch.nan_to_num(nmass, nan=0.0, posinf=0.0, neginf=0.0)
                null_mass_mean = float(nmass.mean().item())
                null_mass_p50 = float(nmass.quantile(0.5).item())
                null_mass_p90 = float(nmass.quantile(0.9).item())
                meta["null_mass_mean"] = null_mass_mean
                meta["null_mass_p50"] = null_mass_p50
                meta["null_mass_p90"] = null_mass_p90
            if isinstance(ctx_payload, dict):
                eff_sample = ctx_payload.get("eff_sample")
                if torch.is_tensor(eff_sample):
                    g_bias_finite_ratio = float(torch.isfinite(eff_sample.detach().float()).float().mean().item())
                base_sample = ctx_payload.get("base_sample")
                if torch.is_tensor(base_sample):
                    base_finite_ratio = float(torch.isfinite(base_sample.detach().float()).float().mean().item())
            finite_checks = []
            for v in (router_prob_finite_ratio, non_null_finite_ratio, null_finite_ratio, g_bias_finite_ratio, base_finite_ratio):
                if math.isfinite(float(v)):
                    finite_checks.append(float(v))
            if finite_checks:
                router_stats_valid = bool(min(finite_checks) >= 0.999999)
            meta["router_prob_finite_ratio"] = float(router_prob_finite_ratio)
            meta["non_null_finite_ratio"] = float(non_null_finite_ratio)
            meta["null_finite_ratio"] = float(null_finite_ratio)
            meta["g_bias_finite_ratio"] = float(g_bias_finite_ratio)
            meta["base_finite_ratio"] = float(base_finite_ratio)
            meta["router_stats_valid"] = bool(router_stats_valid)
            if logit_delta_abs is not None and logit_delta_head_l1 is not None and int(head_limit) > 0:
                meta["logit_delta_l1_mean"] = float(logit_delta_abs.mean().item())
                meta["logit_delta_linf"] = float(logit_delta_abs.max().item())
                meta["logit_delta_head_l1_mean"] = [float(x.item()) for x in logit_delta_head_l1]
            if output_delta_stats is not None:
                meta.update(output_delta_stats)
            with (out_root / "meta.json").open("w", encoding="utf-8") as fout:
                json.dump(meta, fout, ensure_ascii=False, indent=2)

            self._export_eval_attn_pattern_features(
                out_root=out_root,
                model_type=str(
                    getattr(self, "eval_attn_pattern_feature_model_type", "")
                    or getattr(self.config, "eval_attn_pattern_feature_model_type", "")
                ),
                layer_idx=int(lid),
                case_id=int(case_id),
                step_val=int(step_val),
                q_take=int(q_take),
                k_take=int(k_take),
                p_base_cpu=p_base_cpu[:head_limit],
                p_wav_cpu=p_wav_cpu[:head_limit],
                zb_cpu=(zb_cpu[:head_limit] if zb_cpu is not None else None),
                zw_cpu=(zw_cpu[:head_limit] if zw_cpu is not None else None),
                q_repr=q_repr,
                k_repr=k_repr,
                batch_index=int(bidx),
                wavmass_mean=wavmass_mean,
                wavmass_p50=wavmass_p50,
                wavmass_p90=wavmass_p90,
                null_mass_mean=null_mass_mean,
                null_mass_p50=null_mass_p50,
                null_mass_p90=null_mass_p90,
                router_prob_finite_ratio=router_prob_finite_ratio,
                non_null_finite_ratio=non_null_finite_ratio,
                null_finite_ratio=null_finite_ratio,
                g_bias_finite_ratio=g_bias_finite_ratio,
                base_finite_ratio=base_finite_ratio,
                router_stats_valid=router_stats_valid,
            )

            if bool(getattr(self, "eval_attn_topk_enabled", True)):
                self._export_eval_attn_topk_table(
                    out_root=out_root,
                    layer_idx=int(lid),
                    case_id=int(case_id),
                    step_val=int(step_val),
                    head_limit=int(head_limit),
                    q_take=int(q_take),
                    k_take=int(k_take),
                    p_base_cpu=p_base_cpu[:head_limit],
                    p_wav_cpu=p_wav_cpu[:head_limit],
                    logit_delta_cpu=(logit_delta_cpu[:head_limit] if logit_delta_cpu is not None else None),
                    input_ids=input_ids,
                    batch_index=int(bidx),
                )

            if bool(self.eval_attn_heatmap_save_pt):
                save_reduced_only = bool(getattr(self, "eval_attn_heatmap_save_pt_reduced_only", False))
                pt_resize_to = int(getattr(self, "eval_attn_heatmap_pt_resize_to", 0))

                payload = {
                    "meta": meta,
                    "p_base": _masked_reduce_for_pt_export(p_base_cpu, is_logit=False),
                    "p_wavelet": _masked_reduce_for_pt_export(p_wav_cpu, is_logit=False),
                    "p_delta": _masked_reduce_for_pt_export(p_delta_cpu, is_logit=False),
                }
                payload["meta"]["save_pt_reduced_only"] = save_reduced_only
                payload["meta"]["pt_resize_to"] = int(pt_resize_to)
                payload["meta"]["finite_fraction_p_base"] = float(torch.isfinite(p_base_cpu[:head_limit]).float().mean().item())
                payload["meta"]["finite_fraction_p_wavelet"] = float(torch.isfinite(p_wav_cpu[:head_limit]).float().mean().item())
                payload["meta"]["finite_fraction_p_delta"] = float(torch.isfinite(p_delta_cpu[:head_limit]).float().mean().item())
                payload["meta"]["analysis_safe_export"] = True
                # Save logits whenever they are available so downstream analysis can
                # compare probability-space and logit-space behavior from one .pt.
                if logit_delta_cpu is not None:
                    payload["logit_base"] = _masked_reduce_for_pt_export(zb_cpu, is_logit=True)
                    payload["logit_wavelet"] = _masked_reduce_for_pt_export(zw_cpu, is_logit=True)
                    payload["logit_delta"] = _masked_reduce_for_pt_export(logit_delta_cpu, is_logit=False)
                    payload["meta"]["finite_fraction_logit_base"] = float(torch.isfinite(zb_cpu[:head_limit]).float().mean().item())
                    payload["meta"]["finite_fraction_logit_wavelet"] = float(torch.isfinite(zw_cpu[:head_limit]).float().mean().item())
                    payload["meta"]["finite_fraction_logit_delta"] = float(torch.isfinite(logit_delta_cpu[:head_limit]).float().mean().item())
                if (
                    bool(getattr(self, "eval_attn_heatmap_save_pt_outputs", False))
                    and (ob_cpu is not None)
                    and (ow_cpu is not None)
                    and (od_cpu is not None)
                    and (not save_reduced_only)
                ):
                    h_take = min(int(head_limit), int(ob_cpu.shape[1]))
                    payload["out_base"] = ob_cpu[:, :h_take, :]
                    payload["out_wavelet"] = ow_cpu[:, :h_take, :]
                    payload["out_delta"] = od_cpu[:, :h_take, :]
                torch.save(payload, out_root / "attn_probs.pt")

            if bool(self.eval_attn_heatmap_save_png):
                q_vis_start = min(max(int(self.eval_attn_heatmap_min_valid_keys) - 1, 0), max(q_take - 1, 0))
                vmax_q = float(max(0.0, min(1.0, float(self.eval_attn_heatmap_vmax_quantile))))
                d_q = float(max(0.0, min(1.0, float(self.eval_attn_heatmap_delta_quantile))))
                ld_q = float(max(0.0, min(1.0, float(self.eval_attn_heatmap_logit_delta_quantile))))
                show_logit_delta = bool(self.eval_attn_heatmap_logit_delta_png) and (logit_delta_cpu is not None)
                for hid in range(int(head_limit)):
                    pb = p_base_cpu[hid]
                    pw = p_wav_cpu[hid]
                    pd = p_delta_cpu[hid]
                    pb_ref = pb[q_vis_start:, :]
                    pw_ref = pw[q_vis_start:, :]
                    pd_ref = pd[q_vis_start:, :].abs()
                    vmax = max(
                        self._heatmap_quantile_or_max(pb_ref, quantile=vmax_q, min_value=1e-8),
                        self._heatmap_quantile_or_max(pw_ref, quantile=vmax_q, min_value=1e-8),
                        1e-8,
                    )
                    dabs = self._heatmap_quantile_or_max(pd_ref, quantile=d_q, min_value=1e-8)
                    if show_logit_delta:
                        ld = logit_delta_cpu[hid]
                        ld_ref = ld[q_vis_start:, :].abs()
                        ldabs = self._heatmap_quantile_or_max(ld_ref, quantile=ld_q, min_value=1e-8)
                        fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
                    else:
                        ld = None
                        ldabs = None
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
                    im0 = axes[0].imshow(
                        pb.numpy(),
                        cmap=self.eval_attn_heatmap_cmap,
                        vmin=0.0,
                        vmax=vmax,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    im1 = axes[1].imshow(
                        pw.numpy(),
                        cmap=self.eval_attn_heatmap_cmap,
                        vmin=0.0,
                        vmax=vmax,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    im2 = axes[2].imshow(
                        pd.numpy(),
                        cmap=self.eval_attn_heatmap_delta_cmap,
                        vmin=-dabs,
                        vmax=dabs,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    im3 = None
                    if show_logit_delta and ld is not None and ldabs is not None:
                        im3 = axes[3].imshow(
                            ld.numpy(),
                            cmap=self.eval_attn_heatmap_logit_delta_cmap,
                            vmin=-ldabs,
                            vmax=ldabs,
                            aspect="auto",
                            interpolation="nearest",
                        )
                    if bool(getattr(self, "eval_attn_heatmap_show_colorbar", True)):
                        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                        if im3 is not None:
                            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
                    axes[0].set_title("base_softmax")
                    axes[1].set_title("wavelet_softmax")
                    axes[2].set_title("wavelet-base")
                    if show_logit_delta:
                        axes[3].set_title("logit_delta")
                    xtick_stride = int(getattr(self, "eval_attn_heatmap_xtick_stride", 512))
                    xticks = []
                    if xtick_stride > 0:
                        xticks = list(range(0, int(k_take), xtick_stride))
                    if int(k_take) > 1:
                        last_x = int(k_take) - 1
                        if last_x not in xticks:
                            xticks.append(last_x)
                    for ax in axes:
                        if len(xticks) > 0:
                            ax.set_xticks(xticks)
                            ax.set_xticklabels([str(x) for x in xticks], fontsize=7)
                            ax.set_xlabel("key position", fontsize=8)
                        else:
                            ax.set_xticks([])
                        ax.set_yticks([])
                    fig.suptitle(
                        (
                            f"layer={lid} head={hid} step={step_val} "
                            f"wavelet={int(bool(has_wavelet))} rel={int(bool(rel_applied))} "
                            f"q_len={int(q_take)} k_len={int(k_take)}"
                        ),
                        fontsize=9,
                    )
                    fig.savefig(out_root / f"head{hid:02d}.png", dpi=int(self.eval_attn_heatmap_dpi))
                    plt.close(fig)

            self._eval_attn_heatmap_export_count += 1
            self._eval_attn_heatmap_emit(
                (
                    f"[EvalAttnHeatmap] layer={lid} case={case_id} heads={head_limit}/{H} "
                    f"q={q_take} k={k_take} rel={int(bool(rel_applied))} "
                    f"delta_l1={meta['delta_l1_mean']:.3e} delta_linf={meta['delta_linf']:.3e} "
                    f"logit_l1={float(meta.get('logit_delta_l1_mean', 0.0)):.3e} "
                    f"out={str(out_root)}"
                )
            )

            _stop_layer_check = self.eval_attn_heatmap_stop_layer
            if _stop_layer_check is None:
                _cfg_for_stop = getattr(self, "config", None)
                _total_for_stop = self._infer_total_layers_from_config(_cfg_for_stop)
                if _total_for_stop is not None:
                    _stop_layer_check = int(_total_for_stop) - 1
            _at_last_layer = (_stop_layer_check is not None) and (int(lid) == int(_stop_layer_check))
            if _at_last_layer:
                import gc as _gc; _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if bool(getattr(self, "eval_attn_heatmap_stop_after_case", False)) and (
                int(self._eval_attn_heatmap_export_count) >= int(self.eval_attn_heatmap_case_limit)
            ):
                if _at_last_layer:
                    self._eval_attn_heatmap_emit(
                        f"[EvalAttnHeatmap] stop_after_case=1 case_limit={int(self.eval_attn_heatmap_case_limit)} reached at layer={lid}; exiting."
                    )
                    os._exit(0)
        except Exception as exc:
            self._eval_attn_heatmap_emit(
                f"[EvalAttnHeatmap][warn] layer={lid} case={case_id} export_failed=1 err={str(exc)}"
            )

    def _wavelet_export_kind(self) -> str:
        if bool(getattr(self, "wavelet_viz_export", False)):
            return "viz"
        if bool(getattr(self, "wavelet_analysis_export", False)):
            return "analysis"
        return "off"

    def _wavelet_analysis_enabled_for_layer(self, layer_idx: Optional[int], need_log: bool = False) -> bool:
        export_kind = self._wavelet_export_kind()
        if export_kind == "off":
            return False
        # New viz export is eval-only by design.
        if export_kind == "viz" and self.training:
            return False
        if (not need_log) and self.training:
            return False
        if not self._wavelet_analysis_rank0():
            return False
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        if not self._rel_layer_enabled(lid, config=self.config):
            return False
        if export_kind == "viz":
            cap = int(getattr(self, "wavelet_viz_max_batches", 0))
            out_dir = str(getattr(self, "wavelet_viz_outdir", "")).strip()
        else:
            cap = int(getattr(self, "wavelet_analysis_max_batches", 0))
            out_dir = str(getattr(self, "wavelet_analysis_output_dir", "")).strip()
        if cap <= 0:
            return False
        if int(self._wavelet_analysis_layer_counts.get(lid, 0)) >= cap:
            return False
        return len(out_dir) > 0

    def _wavelet_analysis_step(self, layer_idx: Optional[int], step: Optional[int]) -> int:
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        step_val = self._to_int_or_none(step)
        if step_val is not None:
            return int(step_val)
        local = int(self._wavelet_analysis_layer_steps.get(lid, 0)) + 1
        self._wavelet_analysis_layer_steps[lid] = local
        return local

    def _wavelet_analysis_write_record(self, layer_idx: Optional[int], step: Optional[int], payload: dict):
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        export_kind = self._wavelet_export_kind()
        if export_kind == "off":
            return
        if not self._wavelet_analysis_enabled_for_layer(layer_idx=lid, need_log=False):
            return
        if export_kind == "viz":
            out_root = Path(str(getattr(self, "wavelet_viz_outdir", "")).strip())
            run_tag = str(getattr(self, "wavelet_viz_run_tag", "default")).strip() or "default"
            model_size = str(getattr(self, "wavelet_viz_model_size", "unknown")).strip() or "unknown"
            mode_name = str(getattr(self, "wavelet_viz_mode", getattr(self, "wavelet_mode", "unknown")))
            seed_val = int(getattr(self, "wavelet_viz_seed", getattr(self, "wavelet_analysis_seed", -1)))
            out_file = out_root / "wavelet_viz" / model_size / run_tag / f"layer{lid:02d}.jsonl"
        else:
            out_root = Path(str(getattr(self, "wavelet_analysis_output_dir", "")).strip())
            run_tag = str(getattr(self, "wavelet_analysis_run_tag", "default")).strip() or "default"
            model_size = str(getattr(self, "wavelet_viz_model_size", "unknown")).strip() or "unknown"
            mode_name = str(getattr(self, "wavelet_analysis_mode", "unknown"))
            seed_val = int(getattr(self, "wavelet_analysis_seed", -1))
            out_file = out_root / "wavelet_analysis" / run_tag / f"layer{lid:02d}.jsonl"
        step_val = self._wavelet_analysis_step(layer_idx=lid, step=step)
        cfg = getattr(self, "config", None)
        dataset_cfg = "unknown"
        block_size = 0
        bucket_size = 0
        if cfg is not None:
            if export_kind == "viz":
                dataset_cfg = str(getattr(cfg, "wavelet_viz_dataset_config", getattr(cfg, "wavelet_analysis_dataset_config", "unknown")))
                block_size = int(getattr(cfg, "wavelet_viz_block_size", getattr(cfg, "wavelet_analysis_block_size", getattr(cfg, "block_size", 0))))
                bucket_size = int(getattr(cfg, "wavelet_viz_bucket_size", getattr(cfg, "wavelet_analysis_bucket_size", getattr(cfg, "xsum_bucket_size", 0))))
            else:
                dataset_cfg = str(getattr(cfg, "wavelet_analysis_dataset_config", "unknown"))
                block_size = int(getattr(cfg, "wavelet_analysis_block_size", getattr(cfg, "block_size", 0)))
                bucket_size = int(getattr(cfg, "wavelet_analysis_bucket_size", getattr(cfg, "xsum_bucket_size", 0)))
        bucket_t = int(payload.get("bucket_T", payload.get("seq_len", block_size if block_size > 0 else 0)))
        rec = {
            "run_tag": run_tag,
            "mode": mode_name,
            "model_size": model_size,
            "seed": seed_val,
            "dataset": str(getattr(cfg, "dataset_name", "unknown")) if cfg is not None else "unknown",
            "dataset_config": dataset_cfg,
            "block_size": int(block_size),
            "bucket_size": int(bucket_size),
            "bucket_T": int(bucket_t),
            "layer": int(lid),
            "step": int(step_val),
        }
        rec.update(payload)
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("a", encoding="utf-8") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._wavelet_analysis_layer_counts[lid] = int(self._wavelet_analysis_layer_counts.get(lid, 0)) + 1
        except Exception as exc:
            if lid not in self._wavelet_analysis_warned:
                self._wavelet_analysis_warned.add(lid)
                self._k1_emit_log(
                    f"[wavelet_export warn] kind={export_kind} layer={lid} write_failed=1 err={str(exc)} out={str(out_file)}"
                )

    def _wavelet_ctxscale_current_scales(self, device=None, dtype=torch.float32) -> torch.Tensor:
        if getattr(self, "wavelet_ctxscale_learnable_scale", False):
            scales = 2.0 ** self.wavelet_ctxscale_scale_exp * SCALE_MULTIPLIER_DICT[self.bias_type]
        else:
            scales = self.wavelet_ctxscale_scales
        if device is not None:
            scales = scales.to(device=device, dtype=dtype)
        else:
            scales = scales.to(dtype=dtype)
        return scales

    def _wavelet_analysis_emit_pa_baseline(self, *, layer_idx: Optional[int], step: Optional[int], T: int, B: int):
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        if not self._wavelet_analysis_enabled_for_layer(layer_idx=lid, need_log=False):
            return
        scales = self._wavelet_ctxscale_current_scales().detach()
        K = int(scales.numel())
        zeros = [0.0 for _ in range(K)]
        pi_pa = [1.0] + [0.0 for _ in range(K)]
        self._wavelet_analysis_write_record(
            layer_idx=lid,
            step=step,
            payload={
                "wavelet_mode": "off",
                "basis_control": "none",
                "scale_values": [float(x.item()) for x in scales],
                "scales": [float(x.item()) for x in scales],
                "K": int(K),
                "pi_mean": pi_pa,
                "energy_i": zeros,
                "absmean_i": zeros,
                "bias_energy_per_scale": zeros,
                "bias_absmean_per_scale": zeros,
                "width_mean_i": zeros,
                "width_p50_i": zeros,
                "width_p90_i": zeros,
                "width_mean_per_scale": zeros,
                "width_p50_per_scale": zeros,
                "width_p90_per_scale": zeros,
                "pi_entropy_mean": 0.0,
                "pi_entropy_p50": 0.0,
                "pi_entropy_p90": 0.0,
                "pi_top1_mean": 1.0,
                "pi_top1_p90": 1.0,
                "pi_margin_mean": 1.0,
                "pi_margin_p50": 1.0,
                "pi_margin_p90": 1.0,
                "pi_null_mean": 1.0,
                "null_mean": 1.0,
                "sigma_mean": 0.0,
                "sigma_std": 0.0,
                "flip_probability_estimate": 0.0,
                "beta_over_T_p50": 0.0,
                "beta_over_T_p90": 0.0,
                "beta_over_T_p99": 0.0,
                "beta_over_s_top1_p50": 0.0,
                "beta_over_s_top1_p90": 0.0,
                "beta_over_s_top1_p99": 0.0,
                "beta_over_s_exp_p50": 0.0,
                "beta_over_s_exp_p90": 0.0,
                "beta_over_s_exp_p99": 0.0,
                "g_layer": 0.0,
                "g_layer_raw": 0.0,
                "grad_abs": float("nan"),
                "sat_low": 0,
                "sat_high": 0,
                "sat_extreme": 0,
                "clamp_frac": 0.0,
                "eff_bias_abs_p99": 0.0,
                "q_sample_count": int(0),
                "qk_sample_count": int(0),
                "batch_size": int(B),
                "seq_len": int(T),
                "bucket_T": int(T),
            },
        )

    def _ensure_wavelet_gate_grad_hook(self):
        # from_pretrained/DDP can replace Parameter objects; keep hook bound to current gate parameter.
        p = getattr(self, "wavelet_logit_bias_a", None)
        if p is None or (not torch.is_tensor(p)) or (not bool(getattr(p, "requires_grad", False))):
            return
        cur_id = id(p)
        prev_id = getattr(self, "_wavelet_gate_grad_hook_param_id", None)
        handle = getattr(self, "_wavelet_gate_grad_hook_handle", None)
        hook_missing = False
        try:
            hooks = getattr(p, "_backward_hooks", None)
            hook_missing = hooks is not None and len(hooks) == 0
        except Exception:
            hook_missing = False
        if prev_id == cur_id and handle is not None and (not hook_missing):
            return
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
        self._wavelet_gate_grad_hook_handle = None
        try:
            self._wavelet_gate_grad_hook_handle = p.register_hook(self._capture_wavelet_gate_grad)
        except Exception:
            self._wavelet_gate_grad_hook_handle = None
        self._wavelet_gate_grad_hook_param_id = cur_id

    def _capture_wavelet_gate_grad(self, grad: torch.Tensor):
        self._wavelet_gate_grad_seen = True
        grad_abs = float("nan")
        grad_p50 = float("nan")
        grad_p90 = float("nan")
        grad_max = float("nan")
        grad_zero_ratio = float("nan")
        grad_finite_ratio = float("nan")
        grad_nonfinite = -1
        grad_ret = grad
        try:
            if grad is not None:
                gf = grad.detach().to(dtype=torch.float32)
                n_all = int(gf.numel())
                if n_all == 0:
                    grad_abs = 0.0
                    grad_p50 = 0.0
                    grad_p90 = 0.0
                    grad_max = 0.0
                    grad_zero_ratio = 1.0
                    grad_finite_ratio = 1.0
                    grad_nonfinite = 0
                    grad_clean = gf
                else:
                    finite_mask = torch.isfinite(gf)
                    n_finite = int(finite_mask.sum().item())
                    grad_nonfinite = int(n_all - n_finite)
                    grad_finite_ratio = float(n_finite / float(max(1, n_all)))
                    grad_clean = torch.nan_to_num(gf, nan=0.0, posinf=0.0, neginf=0.0)
                    clip_v = float(getattr(self, "wavelet_gate_grad_clip", 0.0))
                    if clip_v > 0.0:
                        grad_clean = grad_clean.clamp(min=-clip_v, max=clip_v)
                    g_abs = grad_clean.abs().reshape(-1)
                    grad_abs = float(g_abs.mean().item())
                    grad_max = float(g_abs.max().item())
                    qv = _quantiles_flat(g_abs, qs=(0.5, 0.9))
                    grad_p50 = float(qv["p50"])
                    grad_p90 = float(qv["p90"])
                    grad_zero_ratio = float((g_abs == 0).to(dtype=torch.float32).mean().item())
                grad_ret = grad_clean.to(dtype=grad.dtype, device=grad.device)
        except Exception:
            grad_abs = float("nan")
            grad_p50 = float("nan")
            grad_p90 = float("nan")
            grad_max = float("nan")
            grad_zero_ratio = float("nan")
            grad_finite_ratio = float("nan")
            grad_nonfinite = -1
            grad_ret = grad
        self._wavelet_gate_last_grad_abs = grad_abs
        self._wavelet_gate_last_grad_p50 = grad_p50
        self._wavelet_gate_last_grad_p90 = grad_p90
        self._wavelet_gate_last_grad_max = grad_max
        self._wavelet_gate_last_grad_zero_ratio = grad_zero_ratio
        self._wavelet_gate_last_grad_finite_ratio = grad_finite_ratio
        self._wavelet_gate_last_grad_nonfinite = grad_nonfinite
        return grad_ret

    def _resolve_wavelet_gate_step(self, step) -> int:
        step_val = self._to_int_or_none(step)
        if step_val is None:
            cfg = getattr(self, "config", None)
            if cfg is not None:
                step_val = self._to_int_or_none(getattr(cfg, "router_global_step", None))
        if step_val is None:
            self._wavelet_gate_local_step += 1
            step_val = int(self._wavelet_gate_local_step)
        return int(step_val)

    def _ctxscale_gate_state(self, *, step, g_max: float, g_layer_raw: torch.Tensor):
        step_val = self._resolve_wavelet_gate_step(step)
        self._ensure_wavelet_gate_grad_hook()
        if self._wavelet_gate_last_metrics_step == step_val and self._wavelet_gate_last_metrics is not None:
            return dict(self._wavelet_gate_last_metrics)

        a_raw_val = float(g_layer_raw.detach().to(dtype=torch.float32).item())
        if a_raw_val >= 0.0:
            sig = 1.0 / (1.0 + math.exp(-min(a_raw_val, 80.0)))
        else:
            expv = math.exp(max(a_raw_val, -80.0))
            sig = expv / (1.0 + expv)
        sig = float(sig)
        g_val = float(g_max) * sig
        sat_low = int(g_val < 0.01 * float(g_max))
        sat_high = int(g_val > 0.99 * float(g_max))
        sat_extreme = int(sat_low or sat_high)

        grad_abs = float("nan")
        grad_p50 = float("nan")
        grad_p90 = float("nan")
        grad_max = float("nan")
        grad_zero_ratio = float("nan")
        grad_finite_ratio = float("nan")
        grad_nonfinite = -1
        if self._wavelet_gate_last_grad_abs is not None:
            try:
                grad_abs = float(self._wavelet_gate_last_grad_abs)
            except Exception:
                grad_abs = float("nan")
        if self._wavelet_gate_last_grad_finite_ratio is not None:
            try:
                grad_finite_ratio = float(self._wavelet_gate_last_grad_finite_ratio)
            except Exception:
                grad_finite_ratio = float("nan")
        if self._wavelet_gate_last_grad_p50 is not None:
            try:
                grad_p50 = float(self._wavelet_gate_last_grad_p50)
            except Exception:
                grad_p50 = float("nan")
        if self._wavelet_gate_last_grad_p90 is not None:
            try:
                grad_p90 = float(self._wavelet_gate_last_grad_p90)
            except Exception:
                grad_p90 = float("nan")
        if self._wavelet_gate_last_grad_max is not None:
            try:
                grad_max = float(self._wavelet_gate_last_grad_max)
            except Exception:
                grad_max = float("nan")
        if self._wavelet_gate_last_grad_zero_ratio is not None:
            try:
                grad_zero_ratio = float(self._wavelet_gate_last_grad_zero_ratio)
            except Exception:
                grad_zero_ratio = float("nan")
        if self._wavelet_gate_last_grad_nonfinite is not None:
            try:
                grad_nonfinite = int(self._wavelet_gate_last_grad_nonfinite)
            except Exception:
                grad_nonfinite = -1
        if math.isfinite(grad_abs):
            grad_abs = abs(grad_abs)
        else:
            grad_abs = float("nan")
        grad_missing = int(grad_nonfinite < 0)
        grad_zero = int(grad_abs == 0.0) if (math.isfinite(grad_abs) and grad_nonfinite == 0) else 0

        delta_a = float("nan")
        update_ratio = float("nan")
        if self._wavelet_gate_prev_a is not None and self._wavelet_gate_prev_step != step_val:
            prev_a = float(self._wavelet_gate_prev_a)
            delta_a = float(a_raw_val - prev_a)
            update_ratio = float(abs(delta_a) / (abs(prev_a) + float(self.wavelet_ctxscale_lock_update_eps)))

        if self._wavelet_gate_prev_step != step_val:
            self._wavelet_gate_prev_a = float(a_raw_val)
            self._wavelet_gate_prev_step = int(step_val)
            self._wavelet_gate_hist.append(
                {
                    "sat_extreme": int(sat_extreme),
                    "grad_abs_p50": float(grad_p50),
                    "update_ratio": float(update_ratio),
                }
            )

        locked = bool(self._wavelet_gate_locked)
        if (not locked) and len(self._wavelet_gate_hist) >= int(self.wavelet_ctxscale_lock_window):
            hist = list(self._wavelet_gate_hist)
            n = max(1, len(hist))
            sat_ratio = sum(int(item.get("sat_extreme", 0)) for item in hist) / float(n)
            grad_vals = [
                float(item.get("grad_abs_p50", float("nan")))
                for item in hist
                if math.isfinite(float(item.get("grad_abs_p50", float("nan"))))
            ]
            upd_vals = [
                float(item.get("update_ratio", float("nan")))
                for item in hist
                if math.isfinite(float(item.get("update_ratio", float("nan"))))
            ]
            grad_med = float("inf")
            upd_med = float("inf")
            if len(grad_vals) > 0:
                grad_med = float(torch.tensor(grad_vals, dtype=torch.float32).median().item())
            if len(upd_vals) > 0:
                upd_med = float(torch.tensor(upd_vals, dtype=torch.float32).median().item())
            if (
                sat_ratio > 0.7
                and grad_med < float(self.wavelet_ctxscale_lock_grad_eps)
                and upd_med < float(self.wavelet_ctxscale_lock_update_eps)
            ):
                locked = True
                self._wavelet_gate_locked = True
                self._k1_emit_log(
                    f"[wavelet gate lock] layer={int(self.layer_idx)} step={int(step_val)} "
                    f"sat_low={int(sat_low)} sat_high={int(sat_high)} "
                    f"grad_abs={float(grad_abs):.6e} grad_finite={float(grad_finite_ratio):.6e} "
                    f"grad_nf={int(grad_nonfinite)} update_ratio={float(update_ratio):.6e}"
                )

        metrics = {
            "step": int(step_val),
            "g_layer_raw": float(a_raw_val),
            "sig": float(sig),
            "g_layer_pre": float(g_val),
            "sat_low": int(sat_low),
            "sat_high": int(sat_high),
            "sat_extreme": int(sat_extreme),
            "grad_abs": float(grad_abs),
            "grad_abs_p50": float(grad_p50),
            "grad_abs_p90": float(grad_p90),
            "grad_abs_max": float(grad_max),
            "grad_zero_ratio": float(grad_zero_ratio),
            "grad_finite_ratio": float(grad_finite_ratio),
            "grad_nonfinite": int(grad_nonfinite),
            "grad_missing": int(grad_missing),
            "grad_zero": int(grad_zero),
            "delta_a": float(delta_a),
            "update_ratio": float(update_ratio),
            "locked": int(locked),
        }
        if int(grad_missing) == 1 and self._wavelet_gate_last_missing_warn_step != int(step_val):
            self._wavelet_gate_last_missing_warn_step = int(step_val)
            self._k1_emit_log(
                f"[wavelet gate grad missing] layer={int(self.layer_idx)} step={int(step_val)} "
                f"req_grad={int(bool(self.wavelet_logit_bias_a.requires_grad))} "
                f"grad_en={int(bool(torch.is_grad_enabled()))} "
                f"sat_extreme={int(sat_extreme)} g_raw={float(a_raw_val):.6e}"
            )
        if int(grad_nonfinite) > 0 and self._wavelet_gate_last_nf_warn_step != int(step_val):
            self._wavelet_gate_last_nf_warn_step = int(step_val)
            self._k1_emit_log(
                f"[wavelet gate grad nf] layer={int(self.layer_idx)} step={int(step_val)} "
                f"grad_nf={int(grad_nonfinite)} grad_finite={float(grad_finite_ratio):.6e} "
                f"grad_abs={float(grad_abs):.6e} sat_extreme={int(sat_extreme)} "
                f"g_raw={float(a_raw_val):.6e}"
            )
        self._wavelet_gate_last_metrics_step = int(step_val)
        self._wavelet_gate_last_metrics = dict(metrics)
        return metrics

    @staticmethod
    def rms_normalize(x: torch.Tensor, eps: float = 1e-6):
        xf = x.to(dtype=torch.float32)
        rms = torch.sqrt(xf.pow(2).mean().clamp_min(0.0) + float(eps))
        return xf / rms

    def _logit_bias_should_log(self, *, global_step=None, config=None) -> tuple[bool, int]:
        step_val = self._to_int_or_none(global_step)
        if step_val is None and config is not None:
            step_val = self._to_int_or_none(getattr(config, "router_global_step", None))
        if step_val is None:
            self.wavelet_logit_bias_local_step += 1
            step_val = int(self.wavelet_logit_bias_local_step)
        log_every = None
        if config is not None:
            log_every = self._to_int_or_none(getattr(config, "wavelet_logit_bias_log_every", None))
            if log_every is None:
                log_every = self._to_int_or_none(getattr(config, "router_log_every", 500))
        if log_every is None:
            log_every = 500
        should_log = (log_every > 0 and step_val >= 0 and (step_val % log_every == 0))
        return bool(should_log), int(step_val)

    def _get_delta_index_matrix(self, T: int, device: torch.device):
        key = (int(T), str(device))
        idx = self._wavelet_delta_index_cache.get(key, None)
        if idx is None or idx.device != device:
            q_pos = torch.arange(T, device=device)
            k_pos = torch.arange(T, device=device)
            delta = k_pos[None, :] - q_pos[:, None]  # n - m
            idx = (delta + (T - 1)).to(dtype=torch.long)
            self._wavelet_delta_index_cache[key] = idx
        return idx

    def compute_wavelet_bias_delta_table(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        wavelet_dtt: torch.Tensor,
    ):
        """
        Build delta lookup table for delta=(key_pos-query_pos)=n-m in [-(T-1), ..., T-1].
        """
        assert wavelet_dtt.dim() == 3, f"expected [D,T,T], got {tuple(wavelet_dtt.shape)}"
        assert int(wavelet_dtt.shape[1]) == int(T) and int(wavelet_dtt.shape[2]) == int(T), (
            tuple(wavelet_dtt.shape),
            int(T),
        )
        wav = wavelet_dtt.to(device=device, dtype=torch.float32).mean(dim=0)  # [T,T]
        delta_idx = self._get_delta_index_matrix(T, device).reshape(-1)  # [T*T]
        val = wav.reshape(-1)
        num_bins = 2 * T - 1
        sum_buf = torch.zeros(num_bins, device=device, dtype=torch.float32)
        cnt_buf = torch.zeros(num_bins, device=device, dtype=torch.float32)
        sum_buf.scatter_add_(0, delta_idx, val)
        cnt_buf.scatter_add_(0, delta_idx, torch.ones_like(val))
        delta_table = sum_buf / cnt_buf.clamp_min(1.0)
        return delta_table.to(dtype=dtype)

    @staticmethod
    def apply_causal_indexing(delta_table: torch.Tensor, delta_index: torch.Tensor, future_mask: torch.Tensor):
        bias = delta_table.index_select(0, delta_index.reshape(-1)).view_as(delta_index)
        return bias.masked_fill(future_mask, 0.0)

    def _build_logit_bias_term(
        self,
        *,
        wavelet_dtt: torch.Tensor,
        T: int,
        future: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype = torch.float32,
    ):
        future_2d = future.view(T, T)
        delta_table_raw = self.compute_wavelet_bias_delta_table(
            T=T,
            device=device,
            dtype=torch.float32,
            wavelet_dtt=wavelet_dtt,
        )
        delta_table_hat = self.rms_normalize(delta_table_raw, eps=float(self.wavelet_logit_bias_eps))
        delta_index = self._get_delta_index_matrix(T, device)
        b_hat = self.apply_causal_indexing(delta_table_hat, delta_index, future_2d)  # [T,T]
        g_layer = F.softplus(self.wavelet_logit_bias_a).to(device=device, dtype=torch.float32)
        eff = g_layer * b_hat
        if self.wavelet_logit_bias_debug_assert:
            if not torch.isfinite(b_hat).all():
                raise FloatingPointError("wavelet_logit_bias: B_hat has non-finite values")
            if not torch.isfinite(eff).all():
                raise FloatingPointError("wavelet_logit_bias: g*B_hat has non-finite values")
        return (
            b_hat.to(dtype=compute_dtype),
            eff.to(dtype=compute_dtype),
            g_layer.to(dtype=compute_dtype),
            future_2d,
        )

    def _log_logit_bias_monitor(
        self,
        *,
        layer_idx: Optional[int],
        step: int,
        g_layer: torch.Tensor,
        b_hat: torch.Tensor,
        eff: torch.Tensor,
        causal_mask_2d: torch.Tensor,
        attn_probs: torch.Tensor,
    ):
        T = int(b_hat.shape[0])
        t_take = max(1, min(int(self.wavelet_logit_bias_log_sample_tokens), T))
        if t_take < T:
            t_idx = torch.linspace(0, T - 1, steps=t_take, device=b_hat.device).long()
            b_eval = b_hat.index_select(0, t_idx).index_select(1, t_idx)
            e_eval = eff.index_select(0, t_idx).index_select(1, t_idx)
            m_eval = causal_mask_2d.index_select(0, t_idx).index_select(1, t_idx)
        else:
            b_eval = b_hat
            e_eval = eff
            m_eval = causal_mask_2d
        valid = ~m_eval
        b_stats = _monitor_flat_stats(b_eval[valid])
        eff_stats = _monitor_flat_stats(e_eval[valid])
        g_stats = _monitor_scalar_stats(g_layer)
        attn_stats = _monitor_attn_prob_stats(
            attn_probs,
            max_queries=int(self.wavelet_logit_bias_log_sample_tokens),
            max_heads=int(self.wavelet_logit_bias_log_sample_heads),
        )
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        msg = (
            f"[wavelet logit_bias stats] layer={lid} step={int(step)} "
            f"g={float(g_layer.detach().float().item()):.6e} "
            f"g_mean={g_stats['mean']:.6e} g_p50={g_stats['p50']:.6e} g_p90={g_stats['p90']:.6e} g_p99={g_stats['p99']:.6e} | "
            f"B_hat mean={b_stats['mean']:.6e} std={b_stats['std']:.6e} abs_p99={b_stats['abs_p99']:.6e} | "
            f"gB_hat mean={eff_stats['mean']:.6e} std={eff_stats['std']:.6e} abs_p99={eff_stats['abs_p99']:.6e} | "
            f"attn_entropy mean={attn_stats['entropy_mean']:.6e} p50={attn_stats['entropy_p50']:.6e} "
            f"p90={attn_stats['entropy_p90']:.6e} p99={attn_stats['entropy_p99']:.6e} | "
            f"attn_top1 mean={attn_stats['top1_mean']:.6e} p50={attn_stats['top1_p50']:.6e} "
            f"p90={attn_stats['top1_p90']:.6e} p99={attn_stats['top1_p99']:.6e} | "
            f"attn_margin mean={attn_stats['margin_mean']:.6e} p50={attn_stats['margin_p50']:.6e} "
            f"p90={attn_stats['margin_p90']:.6e} p99={attn_stats['margin_p99']:.6e}"
        )
        self._k1_emit_log(msg)

    @staticmethod
    def _windowed_mean_sq(x: torch.Tensor, window_cap: int) -> torch.Tensor:
        """Mean of x**2 over the last dim, optionally capped to the first window_cap entries."""
        if window_cap > 0 and window_cap < x.shape[-1]:
            x = x[..., :window_cap]
        return x.pow(2).mean(dim=-1, keepdim=True)

    @staticmethod
    def _rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6, mask: Optional[torch.Tensor] = None):
        xf = x.to(dtype=torch.float32)
        if mask is None:
            denom = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True).clamp_min(0.0) + float(eps))
        else:
            m = mask.to(device=xf.device, dtype=torch.float32)
            while m.dim() < xf.dim():
                m = m.unsqueeze(0)
            count = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
            denom = torch.sqrt((xf.pow(2) * m).sum(dim=-1, keepdim=True).div(count).clamp_min(0.0) + float(eps))
        return xf / denom

    def _rms_norm_wavelet_basis(self, basis_table: torch.Tensor, *, q0: int, eps: float):
        # PAT-225 causal/window-cap ablation: 0 keeps prior full-width RMS behavior.
        basis_table_f = basis_table.to(dtype=torch.float32)
        mean_sq = self._windowed_mean_sq(
            basis_table_f,
            int(getattr(self, "wavelet_ctxscale_rms_train_window", 0)),
        )
        denom = torch.sqrt(mean_sq.clamp_min(0.0) + float(eps))
        return basis_table_f / denom

    @staticmethod
    def _ricker_wavelet(u: torch.Tensor):
        return (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))

    def _apply_ricker_pl4_pattern(
        self,
        basis_table: torch.Tensor,
        base_x: torch.Tensor,
        beta_i: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(getattr(self, "wavelet_ctxscale_pattern_mode", "ricker"))
        if mode == "ricker":
            return basis_table

        scale_f = torch.as_tensor(scale, device=basis_table.device, dtype=torch.float32).clamp_min(1e-12)
        beta = beta_i.unsqueeze(-1)
        out = basis_table
        bin_width = 128.0
        restore_bin = int(getattr(self, "wavelet_ctxscale_restore_bin", -1))

        for bin_idx in range(4):
            if mode == "restore" and bin_idx == restore_bin:
                continue
            left = float(bin_idx) * bin_width
            right = left + bin_width

            mask = (base_x >= left) & (base_x < right)

            left_t = left - beta
            right_t = right - beta
            y_left = self._ricker_wavelet(left_t / scale_f)
            y_right = self._ricker_wavelet(right_t / scale_f)
            alpha = ((base_x - left) / bin_width).clamp(0.0, 1.0)
            linear = y_left + (y_right - y_left) * alpha
            out = torch.where(mask, linear.to(dtype=out.dtype), out)
        return out

    @staticmethod
    def _sine_basis(u: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * u)

    def _morlet_basis(self, u: torch.Tensor) -> torch.Tensor:
        freq = float(getattr(self, "wavelet_morlet_freq", 5.0))
        return torch.exp(-0.5 * u.pow(2)) * torch.cos(freq * u)

    def _ricker_cos_basis(self, u: torch.Tensor) -> torch.Tensor:
        # Same envelope as _ricker_wavelet (and same SCALE_MULTIPLIER_DICT=1.0),
        # just multiplied by a cos carrier -- isolates "does adding oscillation
        # help" from morlet's confound of also silently widening the envelope.
        freq = float(getattr(self, "wavelet_morlet_freq", 5.0))
        return self._ricker_wavelet(u) * torch.cos(freq * u)

    @staticmethod
    def _gaussian_basis(u: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * u.pow(2))

    @staticmethod
    def _linear_basis(
        u: torch.Tensor,
    ) -> torch.Tensor:
        return (1.0 - u.abs() / math.sqrt(3.0)).clamp_min(0.0)

    def _resolve_router_jitter_std(
        self,
        base_std: float,
        *,
        global_step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> float:
        """
        Resolve router jitter target flip ratio.
        The only active control is `router_jitter_flip_ratio` in [0, 0.5).
        Backward compatibility: if absent, fallback to `router_jitter_std`.
        """
        ratio = getattr(self.config, "router_jitter_flip_ratio", None)
        if ratio is None:
            ratio = getattr(self.config, "router_jitter_std", base_std)
        try:
            ratio_f = float(ratio)
        except Exception:
            ratio_f = 0.0
        if not math.isfinite(ratio_f):
            ratio_f = 0.0
        return float(min(max(ratio_f, 0.0), 0.499999))

    def _resolve_wavelet_ctxscale_tau(self, *, global_step: Optional[int] = None) -> float:
        """
        Resolve router temperature for ctxscale branch.

        Supported schedule:
          - `none`: fixed `wavelet_ctxscale_tau`
          - `linear`: `tau_start -> tau_end` over `tau_anneal_steps` after `tau_anneal_warmup`
        """
        base_tau = max(float(getattr(self, "wavelet_ctxscale_tau", 1.0)), 1e-6)
        schedule = str(getattr(self, "wavelet_ctxscale_tau_schedule", "none")).strip().lower()
        if schedule != "linear":
            return base_tau

        tau_start = max(float(getattr(self, "wavelet_ctxscale_tau_start", base_tau)), 1e-6)
        tau_end = max(float(getattr(self, "wavelet_ctxscale_tau_end", base_tau)), 1e-6)
        step_i = self._to_int_or_none(global_step)
        if step_i is None:
            return tau_start

        warmup = max(0, int(getattr(self, "wavelet_ctxscale_tau_anneal_warmup", 0)))
        anneal_steps = int(getattr(self, "wavelet_ctxscale_tau_anneal_steps", 0))
        if anneal_steps <= 0:
            anneal_steps = max(1, int(getattr(self.config, "router_max_steps", 15900)))

        if step_i <= warmup:
            return tau_start

        frac = (float(step_i - warmup) / float(max(1, anneal_steps)))
        frac = min(max(frac, 0.0), 1.0)
        tau = tau_start + (tau_end - tau_start) * frac
        return max(float(tau), 1e-6)

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _add_gaussian_jitter(
        self,
        logits: torch.Tensor,
        std: float,
        *,
        router_name: Optional[str] = None,
        global_step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add gaussian jitter to router logits using only target flip ratio.
        `std` is interpreted as target top1-top2 flip ratio rho0 in (0, 0.5).
        """
        router_name_s = str(router_name) if router_name is not None else "unknown"
        noise_adapt_style_raw = "flipratio_token"
        noise_adapt_style = "flipratio_token"

        if std <= 0:
            self._last_router_jitter_stats = {
                "router_name": router_name_s,
                "style_raw": noise_adapt_style_raw,
                "style_resolved": noise_adapt_style,
                "sigma_mean": 0.0,
                "sigma_std": 0.0,
                "flip_probability_estimate": 0.0,
                "margin_mean": float("nan"),
                "target_flip_probability": float("nan"),
                "injected": 0,
            }
            return logits

        log_router_stats_train = bool(getattr(self.config, "log_train_router_stats", True))
        log_every = int(getattr(self.config, "router_log_every", 500))

        squeeze_head = False
        logits_work = logits
        if logits_work.dim() == 3:
            logits_work = logits_work.unsqueeze(-2)  # [B,T,S] -> [B,T,1,S]
            squeeze_head = True
        elif logits_work.dim() != 4:
            raise ValueError(f"_add_gaussian_jitter expects [B,T,S] or [B,T,H,S], got {tuple(logits.shape)}")

        logits_det = logits_work.detach()
        _, _, H, S = logits_det.shape
        if S < 2:
            self._last_router_jitter_stats = {
                "router_name": router_name_s,
                "style_raw": noise_adapt_style_raw,
                "style_resolved": noise_adapt_style,
                "sigma_mean": 0.0,
                "sigma_std": 0.0,
                "flip_probability_estimate": 0.0,
                "margin_mean": float("nan"),
                "target_flip_probability": float("nan"),
                "injected": 0,
            }
            return logits

        rho0 = min(max(float(std), 1e-6), 0.499999)
        top2 = torch.topk(logits_det, k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).clamp_min(1e-6)  # [B,T,H]
        normal = torch.distributions.Normal(
            loc=logits_det.new_tensor(0.0),
            scale=logits_det.new_tensor(1.0),
        )
        z = normal.icdf(logits_det.new_tensor(rho0)).abs().clamp_min(1e-6)
        denom = math.sqrt(2.0) * z
        sigma_tok = margin / denom
        sigma_eff = sigma_tok.unsqueeze(-1)
        sigma_eff_no_last = sigma_tok
        sigma_eval = torch.nan_to_num(
            sigma_eff_no_last.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(1e-12)
        margin_eval = torch.nan_to_num(margin.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        flip_prob_est = self._normal_cdf(-(margin_eval / (math.sqrt(2.0) * sigma_eval)))
        sigma_flat = sigma_eval.reshape(-1)
        flip_flat = torch.nan_to_num(flip_prob_est, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        margin_flat = margin_eval.reshape(-1)
        sigma_mean = float(sigma_flat.mean().item()) if sigma_flat.numel() > 0 else float("nan")
        sigma_std = float(sigma_flat.std(unbiased=False).item()) if sigma_flat.numel() > 0 else float("nan")
        flip_est_mean = float(flip_flat.mean().item()) if flip_flat.numel() > 0 else float("nan")
        margin_mean = float(margin_flat.mean().item()) if margin_flat.numel() > 0 else float("nan")
        self._last_router_jitter_stats = {
            "router_name": router_name_s,
            "style_raw": noise_adapt_style_raw,
            "style_resolved": noise_adapt_style,
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
            "flip_probability_estimate": flip_est_mean,
            "margin_mean": margin_mean,
            "target_flip_probability": float(rho0),
            "injected": int(self.training),
        }

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
                eff_bt = sigma_eff.detach().reshape(-1, H)
                q = torch.tensor([0.5, 0.9, 0.95], device=eff_bt.device)
                eff_q = torch.quantile(eff_bt, q, dim=0)
                msg = (
                    f"router_name={router_name} "
                    f"[router {'train' if self.training else 'eval'} stats] "
                    f"layer={self.layer_idx} step={global_step}/{max_steps} "
                    f"style={noise_adapt_style_raw} "
                    f"flip_ratio={rho0:.6e} "
                    f"sigma_mean={sigma_mean:.6e} sigma_std={sigma_std:.6e} "
                    f"flip_prob_est={flip_est_mean:.6e} margin_mean={margin_mean:.6e} "
                    f"eff_p50={_fmt_vec(eff_q[0], 6)} eff_p90={_fmt_vec(eff_q[1], 6)} eff_p95={_fmt_vec(eff_q[2], 6)} "
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

        if not self.training:
            if isinstance(self._last_router_jitter_stats, dict):
                self._last_router_jitter_stats["injected"] = 0
            return logits

        noise = torch.randn_like(logits_work) * sigma_eff
        out = logits_work + noise
        if squeeze_head:
            out = out.squeeze(-2)
        if isinstance(self._last_router_jitter_stats, dict):
            self._last_router_jitter_stats["injected"] = 1
        return out

    def _ctxscale_router_feature(self, qf: torch.Tensor, q_corr: torch.Tensor, *, use_mlp: bool = False, hidden_states: Optional[torch.Tensor] = None):
        mode = str(getattr(self, "wavelet_ctx_feat_mode", "q_meanH")).strip().lower()
        feat_ln = self.mlp_bias_ctx_feat_ln if use_mlp else self.wavelet_ctx_feat_ln
        path_ln = self.mlp_bias_ctx_path_ln if use_mlp else self.wavelet_ctx_path_ln
        path_proj = self.mlp_bias_ctx_path_proj if use_mlp else self.wavelet_ctx_path_proj
        ln_dtype = feat_ln.weight.dtype  # match LayerNorm weight dtype (may be bf16 under bf16_full_eval)
        # hidden_ln: route from pre-attention LN-normalized hidden state instead of PaTH q_corr.
        # hidden_states is already ln_1(h^{l-1}) in GPT-2 pre-LN, so no additional LN is applied.
        if mode == "hidden_ln":
            assert hidden_states is not None, "hidden_ln mode requires hidden_states"
            h = hidden_states.detach().to(device=qf.device, dtype=qf.dtype)
            h = h.view(h.shape[0], h.shape[1], self.num_heads, self.head_dim).mean(dim=2)
            return h  # [B, T, head_dim]
        delta = qf - q_corr
        if self.wavelet_ctx_feat_detach_delta:
            delta = delta.detach()
        if getattr(self, "_pat_g0_cap", None) is not None:  # PAT-243 q-q_corr cross-head disagreement probe (default off)
            # Per (batch, position): variance of (q - q_corr) across heads, averaged over
            # head_dim -- a "how much do heads disagree" scalar. The router only ever sees
            # the head-mean (d_mean below), so this measures information the router's own
            # feature discards, to test whether gate strength correlates with head disagreement.
            head_disagreement = delta.var(dim=2, unbiased=False).mean(dim=-1)  # [B, T]
            self._pat_g0_cap.setdefault("qcorr_head_disagreement", []).append(
                (int(self.layer_idx), head_disagreement.detach().float().cpu())
            )
        q_mean = qf.mean(dim=2)
        d_mean = delta.mean(dim=2)
        if getattr(self, "_pat_g0_cap", None) is not None:  # PAT-243: magnitude of the router's actual input feature
            d_mean_norm = d_mean.detach().float().norm(dim=-1)  # [B, T]
            self._pat_g0_cap.setdefault("qcorr_dmean_norm", []).append(
                (int(self.layer_idx), d_mean_norm.cpu())
            )
        if mode == "q_minus_qcorr_meanh":
            return feat_ln(d_mean.to(ln_dtype))
        if mode == "q_minus_qcorr_rmsh":
            d_rms = torch.sqrt(delta.pow(2).mean(dim=2).clamp_min(0.0) + float(self.wavelet_ctx_feat_rms_eps))
            return feat_ln(d_rms.to(ln_dtype))
        if mode == "path_ctx":
            x_cat = torch.cat([q_mean, q_corr.mean(dim=2), d_mean], dim=-1)
            x_cat = path_ln(x_cat.to(ln_dtype))
            return feat_ln(path_proj(x_cat).to(ln_dtype))
        # Default safer baseline: pure query summary.
        return feat_ln(q_mean.to(ln_dtype))

    def _ctxscale_param_count(self, *, use_mlp: bool, include_film: bool) -> int:
        if use_mlp:
            modules = [
                self.mlp_bias_ctx_feat_ln,
                self.mlp_bias_ctx_path_ln,
                self.mlp_bias_ctx_path_proj,
                self.mlp_bias_router,
                self.mlp_bias_shift_ln,
                self.mlp_bias_shift_proj,
                self.mlp_bias_basis_mlp,
                self.mlp_bias_basis_ln,
            ]
            params = [self.mlp_bias_logit_bias_a]
            if self.mlp_bias_logit_bias_a_head is not None:
                params.append(self.mlp_bias_logit_bias_a_head)
            if include_film:
                modules.extend([self.mlp_bias_film_ln, self.mlp_bias_film])
            if hasattr(self, "mlp_bias_param_pad") and self.mlp_bias_param_pad is not None:
                params.append(self.mlp_bias_param_pad)
        else:
            modules = [
                self.wavelet_ctx_feat_ln,
                self.wavelet_ctx_path_ln,
                self.wavelet_ctx_path_proj,
                self.wavelet_ctx_router,
                self.wavelet_shift_ln,
                self.wavelet_shift_proj,
            ]
            params = [self.wavelet_logit_bias_a]
            if self.wavelet_logit_bias_a_head is not None:
                params.append(self.wavelet_logit_bias_a_head)
            if include_film:
                modules.extend([self.wavelet_bias_film_ln, self.wavelet_bias_film])
        total = 0
        for mod in modules:
            total += sum(p.numel() for p in mod.parameters())
        total += sum(p.numel() for p in params)
        return int(total)

    def _wavelet_seed(self) -> int:
        seed = None
        try:
            seed = getattr(self.config, "seed", None)
        except Exception:
            seed = None
        seed_i = self._to_int_or_none(seed)
        if seed_i is None:
            try:
                seed_i = int(torch.initial_seed())
            except Exception:
                seed_i = None
        if seed_i is None:
            seed_i = 0
            if not self._wavelet_basis_seed_warned:
                self._wavelet_basis_seed_warned = True
                self._k1_emit_log("[wavelet basis control warn] missing seed in config and torch; fallback_seed=0")
        return int(seed_i)

    def _wavelet_basis_perm(self, *, K: int, layer_idx: int, device: torch.device) -> torch.Tensor:
        seed_i = self._wavelet_seed()
        key = (seed_i, int(layer_idx), int(K))
        perm_cpu = self._wavelet_basis_perm_cache.get(key, None)
        if perm_cpu is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed_i + 131 * int(layer_idx) + 17 * int(K)))
            perm_cpu = torch.randperm(int(K), generator=gen, device="cpu")
            self._wavelet_basis_perm_cache[key] = perm_cpu
        return perm_cpu.to(device=device)

    def _random_basis_table(
        self,
        *,
        q_len: int,
        T: int,
        scale_idx: int,
        layer_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        seed_i = self._wavelet_seed()
        key = (seed_i, int(layer_idx), int(scale_idx), int(q_len), int(T))
        tab_cpu = self._wavelet_basis_random_cache.get(key, None)
        if tab_cpu is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed_i + 1009 * int(layer_idx) + 101 * int(scale_idx) + 13 * int(q_len) + int(T)))
            tab_cpu = torch.randn((int(q_len), int(T)), generator=gen, dtype=torch.float32, device="cpu")
            self._wavelet_basis_random_cache[key] = tab_cpu
        return tab_cpu.to(device=device, dtype=torch.float32)

    def _ctxscale_apply_bias_film(
        self,
        *,
        bias_chunk: torch.Tensor,
        pi_chunk: torch.Tensor,
        rho_chunk: torch.Tensor,
        g_layer: torch.Tensor,
        use_mlp: bool = False,
    ):
        B, Tq, _ = bias_chunk.shape
        g_ctx = g_layer.reshape(1, 1, 1).expand(B, Tq, 1)
        w_ctx = torch.cat([pi_chunk[..., 1:], rho_chunk.unsqueeze(-1), g_ctx], dim=-1)
        film_ln = self.mlp_bias_film_ln if use_mlp else self.wavelet_bias_film_ln
        film_mod = self.mlp_bias_film if use_mlp else self.wavelet_bias_film
        w_ctx_ln = film_ln(w_ctx)
        film_raw = film_mod(w_ctx_ln)
        s_raw, t_raw = film_raw[..., :1], film_raw[..., 1:]
        clamp_v = float(self.wavelet_ctxscale_film_clamp)
        s_raw = s_raw.clamp(min=-clamp_v, max=clamp_v)
        t_raw = t_raw.clamp(min=-clamp_v, max=clamp_v)
        scale = 1.0 + float(self.wavelet_ctxscale_film_alpha) * torch.tanh(s_raw)
        shift = float(self.wavelet_ctxscale_film_beta) * torch.tanh(t_raw)
        bias_mod = scale * bias_chunk + shift
        return bias_mod, s_raw, t_raw, scale, shift

    def _apply_ctxscale_do_intervention(
        self,
        *,
        pi: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Forward-only intervention hook for ctxscale router probabilities.
        spec format (attached to module as `_ctxscale_do_spec`):
          {
            "enabled": bool,
            "mode": "null" | "small" | "large" | "uniform",
            "target_layer": int
          }
        """
        self._last_ctxscale_do_stat = None
        spec = getattr(self, "_ctxscale_do_spec", None)
        if not isinstance(spec, dict) or (not bool(spec.get("enabled", False))):
            return pi
        if int(spec.get("target_layer", -1)) != int(layer_idx):
            return pi
        mode = str(spec.get("mode", "")).strip().lower()
        if mode not in ("null", "small", "large", "uniform"):
            return pi
        if pi.dim() != 3:
            return pi
        Kp1 = int(pi.shape[-1])
        if Kp1 <= 1:
            return pi
        K = Kp1 - 1

        strict = bool(spec.get("strict", True))
        scale_idx_spec = spec.get("scale_idx", None)
        if scale_idx_spec is None:
            if mode == "small":
                scale_idx = 0
            elif mode == "large":
                scale_idx = K - 1
            else:
                scale_idx = 0
        else:
            scale_idx = max(0, min(K - 1, int(scale_idx_spec)))
        expected_argmax = 0 if mode == "null" else int(scale_idx + 1)

        target_head = spec.get("target_head", None)
        target_heads = spec.get("target_heads", None)
        if target_head is not None or target_heads not in (None, [], (), "all", "*"):
            raise ValueError(
                "Per-head ctxscale intervention has been removed; "
                "only shared-router intervention is supported."
            )

        forced = torch.zeros_like(pi)
        if mode == "null":
            forced[..., 0] = 1.0
        elif mode in ("small", "large"):
            forced[..., int(scale_idx + 1)] = 1.0
        elif mode == "uniform":
            forced[..., 1:] = 1.0 / float(K)

        pi_new = forced

        # validation payload for downstream logging / hard checks.
        try:
            v = pi_new.detach().float()
            target_pi = v
            sum_prob = target_pi.sum(dim=-1)
            pi0 = target_pi[..., 0]
            argmax_idx = target_pi.argmax(dim=-1)
            nnz = (target_pi > 1e-8).sum(dim=-1).float()
            target_scale = target_pi[..., 1:] if K > 0 else target_pi[..., :0]
            uniform_dev = 0.0
            if mode == "uniform" and K > 0 and int(target_scale.numel()) > 0:
                uniform_ref = torch.full_like(target_scale, 1.0 / float(K))
                uniform_dev = float((target_scale - uniform_ref).abs().max().item())

            stat = {
                "enabled": True,
                "mode": mode,
                "layer": int(layer_idx),
                "scope": "shared",
                "strict": int(strict),
                "num_scales": int(K),
                "target_heads": list(range(max(1, int(getattr(self, "num_heads", 1))))),
                "target_head_for_check": 0,
                "scale_idx": int(scale_idx),
                "expected_argmax": int(expected_argmax),
                "pi_dim_in": 3,
                "pi_dim_out": int(pi_new.dim()),
                "target_pi_min": float(target_pi.min().item()),
                "target_pi_max": float(target_pi.max().item()),
                "target_pi0_mean": float(pi0.mean().item()),
                "target_sum_prob_mean": float(sum_prob.mean().item()),
                "target_sum_prob_min": float(sum_prob.min().item()),
                "target_sum_prob_max": float(sum_prob.max().item()),
                "target_argmax_expected_frac": float((argmax_idx == int(expected_argmax)).float().mean().item()),
                "target_nonzero_count_mean": float(nnz.mean().item()),
                "target_nonzero_count_max": float(nnz.max().item()),
                "target_uniform_scale_maxdev": float(uniform_dev),
                "non_target_pi_change_maxabs": 0.0,
                "non_target_pi_change_meanabs": 0.0,
            }

            tol = 1e-6
            if strict:
                if abs(stat["target_sum_prob_mean"] - 1.0) > 1e-6:
                    raise AssertionError(f"do({mode}) failed: target pi sum mean != 1, got {stat['target_sum_prob_mean']:.6e}")
                if abs(stat["target_sum_prob_min"] - 1.0) > 1e-5 or abs(stat["target_sum_prob_max"] - 1.0) > 1e-5:
                    raise AssertionError(
                        f"do({mode}) failed: target pi sum range not ~1 "
                        f"[{stat['target_sum_prob_min']:.6e},{stat['target_sum_prob_max']:.6e}]"
                    )
                if mode == "null":
                    if abs(stat["target_pi0_mean"] - 1.0) > 1e-6:
                        raise AssertionError(f"do(null) failed: pi0 mean={stat['target_pi0_mean']:.6e}")
                    if stat["target_nonzero_count_max"] > 1.0 + tol:
                        raise AssertionError(f"do(null) failed: nonzero_count_max={stat['target_nonzero_count_max']:.6e}")
                elif mode in ("small", "large"):
                    if abs(stat["target_pi0_mean"]) > 1e-6:
                        raise AssertionError(f"do({mode}) failed: pi0 mean={stat['target_pi0_mean']:.6e}")
                    if stat["target_argmax_expected_frac"] < 0.999999:
                        raise AssertionError(
                            f"do({mode}) failed: argmax expected frac={stat['target_argmax_expected_frac']:.6e}, "
                            f"expected_argmax={int(expected_argmax)}"
                        )
                    if stat["target_nonzero_count_max"] > 1.0 + tol:
                        raise AssertionError(f"do({mode}) failed: nonzero_count_max={stat['target_nonzero_count_max']:.6e}")
                elif mode == "uniform":
                    if abs(stat["target_pi0_mean"]) > 1e-6:
                        raise AssertionError(f"do(uniform) failed: pi0 mean={stat['target_pi0_mean']:.6e}")
                    if stat["target_uniform_scale_maxdev"] > 1e-6:
                        raise AssertionError(
                            f"do(uniform) failed: uniform maxdev={stat['target_uniform_scale_maxdev']:.6e}"
                        )

            self._last_ctxscale_do_stat = stat
        except Exception:
            if strict:
                raise
            self._last_ctxscale_do_stat = {
                "enabled": True,
                "mode": mode,
                "layer": int(layer_idx),
                "scope": "validation_failed_non_strict",
            }
        return pi_new

    def _parse_wavelet_intervention_targets(
        self,
        targets_obj,
        *,
        default_layer=None,
        default_heads=None,
    ):
        """Normalize intervention targets into {layer_idx: heads_or_all}."""
        targets = {}

        obj = targets_obj
        if isinstance(obj, str):
            s = obj.strip()
            if s:
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = None

        if isinstance(obj, dict):
            for lk, hv in obj.items():
                try:
                    layer_i = int(lk)
                except Exception:
                    continue
                if isinstance(hv, str) and hv.strip().lower() == "all":
                    targets[layer_i] = "all"
                elif isinstance(hv, int):
                    targets[layer_i] = [int(hv)]
                elif isinstance(hv, (list, tuple, set)):
                    h_list = []
                    for h in hv:
                        try:
                            h_list.append(int(h))
                        except Exception:
                            continue
                    if h_list:
                        targets[layer_i] = sorted(set(h_list))

        if not targets and default_layer is not None:
            try:
                layer_i = int(default_layer)
            except Exception:
                layer_i = None
            if layer_i is not None:
                hv = default_heads
                if isinstance(hv, str):
                    hs = hv.strip().lower()
                    if hs == "all":
                        targets[layer_i] = "all"
                    else:
                        h_list = self._parse_int_list(hv, default=[])
                        if h_list:
                            targets[layer_i] = sorted(set(int(x) for x in h_list))
                elif isinstance(hv, int):
                    targets[layer_i] = [int(hv)]
                elif isinstance(hv, (list, tuple, set)):
                    h_list = []
                    for h in hv:
                        try:
                            h_list.append(int(h))
                        except Exception:
                            continue
                    if h_list:
                        targets[layer_i] = sorted(set(h_list))
        return targets

    def _get_ctxscale_do_spec_for_layer(self, layer_idx: int):
        if not bool(getattr(self, "wavelet_intervention_enable", False)):
            return None
        if str(getattr(self, "wavelet_intervention_mode", "")).strip().lower() != "ctxscale_null":
            return None
        targets = getattr(self, "_wavelet_intervention_targets", None)
        if not isinstance(targets, dict) or not targets:
            return None
        lid = int(layer_idx)
        if lid not in targets:
            return None
        return {
            "enabled": True,
            "mode": "null",
            "target_layer": int(lid),
            "target_heads": targets[lid],
            "strict": bool(getattr(self, "wavelet_intervention_strict", True)),
        }

    def _build_ctxscale_shift_logit_bias_v0(
        self,
        *,
        q: torch.Tensor,
        w: torch.Tensor,
        M_used: torch.Tensor,
        hidden_states: torch.Tensor,
        E_base_raw: torch.Tensor,
        T: int,
        compute_dtype: torch.dtype = torch.float32,
        need_log: bool = False,
        layer_idx: Optional[int] = None,
        step: Optional[int] = None,
        enable_film: bool = False,
        k = None,
    ):
        if hidden_states is None:
            raise ValueError("hidden_states is required for wavelet_mode='logit_bias_ctxscale_shift_v0'")

        B = int(E_base_raw.shape[0])
        device = E_base_raw.device
        eps = float(self.wavelet_logit_bias_eps)
        reg_zero = E_base_raw.new_zeros([])
        self._last_router_entropy_reg_loss = reg_zero
        self._last_router_entropy_reg_active_frac = reg_zero.detach()
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        self._ctxscale_do_spec = self._get_ctxscale_do_spec_for_layer(lid)
        self._last_ctxscale_do_validation = None
        step_val = self._to_int_or_none(step)
        if step_val is None:
            step_val = self._resolve_wavelet_gate_step(None)
        warned_nonfinite = False

        def _warn_nonfinite(name: str):
            nonlocal warned_nonfinite
            if warned_nonfinite:
                return
            warned_nonfinite = True
            msg = f"[wavelet ctxscale_shift_v0 warn] layer={lid} step={int(step_val)} non_finite={name}"
            logger_obj = getattr(self, "logger", None)
            if logger_obj is not None:
                try:
                    logger_obj.warning(msg)
                    return
                except Exception:
                    pass
            print(msg)

        qf = q.to(device=device, dtype=torch.float32)
        wf = w.to(device=device, dtype=torch.float32)
        mf = M_used.to(device=device, dtype=torch.float32)
        scales = self._wavelet_ctxscale_current_scales(device=device, dtype=torch.float32)
        K = int(scales.numel())
        # Expose per-token scale mixture for external analysis (same forward pass).
        self._last_ctxscale_router_prob = None
        self._last_ctxscale_non_null_mass = None
        self._last_ctxscale_null_mass = None
        self._last_ctxscale_monitor_payload = None
        self._last_ctxscale_beta_by_scale = None
        wavelet_mode_resolved = self._normalize_wavelet_mode(getattr(self.config, "wavelet_mode", self.wavelet_mode))
        use_mlp_bias_baseline = bool(wavelet_mode_resolved == "mlp_bias_baseline_v0")
        basis_control = str(getattr(self, "wavelet_basis_control", "none")).strip().lower()
        if basis_control not in ("none", "permute_scales", "random_basis"):
            basis_control = "none"
        if use_mlp_bias_baseline:
            basis_control = "none"

        if use_mlp_bias_baseline and (not bool(getattr(self, "_mlp_bias_param_count_printed", False))) and lid == 0:
            include_film_count = bool(enable_film)
            wavelet_branch_params = self._ctxscale_param_count(use_mlp=False, include_film=include_film_count)
            mlp_branch_params = self._ctxscale_param_count(use_mlp=True, include_film=include_film_count)
            rel_diff = float(abs(float(mlp_branch_params - wavelet_branch_params)) / max(1.0, float(wavelet_branch_params)))
            self._k1_emit_log(
                f"[mlp_bias_baseline_v0 param_count] layer={lid} "
                f"wavelet_branch_params={int(wavelet_branch_params)} "
                f"mlp_branch_params={int(mlp_branch_params)} rel_diff={rel_diff:.6e}"
            )
            self._mlp_bias_param_count_printed = True
            if rel_diff > 0.01:
                raise ValueError(
                    f"mlp_bias_baseline_v0 param mismatch: wavelet={wavelet_branch_params} "
                    f"mlp={mlp_branch_params} rel_diff={rel_diff:.6e}"
                )

        q_corr = torch.einsum("b h t j, b j h d -> b t h d", mf, wf)
        router_mod = self.mlp_bias_router if use_mlp_bias_baseline else self.wavelet_ctx_router
        x_feat = self._ctxscale_router_feature(
            qf,
            q_corr,
            use_mlp=use_mlp_bias_baseline,
            hidden_states=hidden_states,
        )
        if x_feat.dim() != 3:
            raise ValueError(
                "Wavelet router features must have shape [B, T, D]; "
                f"got {tuple(x_feat.shape)}."
            )
        router_logits = router_mod(x_feat)
        if getattr(self, "wavelet_router_length_aware", False) and self.wavelet_ctx_router_length is not None:
            # Broadcast log2(T) (constant within this forward call) as an explicit length
            # feature, added on top of the existing per-query router logits. Lets the
            # router learn a length-conditioned scale preference under mixed-length
            # training instead of only ever seeing content features with no direct T signal.
            _log2_t = torch.log2(torch.tensor(float(T), device=router_logits.device, dtype=torch.float32))
            _len_feat = _log2_t.view(1, 1, 1).to(dtype=self.wavelet_ctx_router_length.weight.dtype)
            _len_bias = self.wavelet_ctx_router_length(_len_feat)  # [1, 1, K+1]
            router_logits = router_logits + _len_bias.to(dtype=router_logits.dtype)
        if getattr(self, "wavelet_router_cosine", False):
            # PAT-244 unified cosine router (CLIP / QK-norm template): gauge-fix EVERY
            # logit (null + all K scales) by L2-normalizing the router feature and each
            # router-weight row, so logit_j = cos(x_feat, w_j) in [-1,1]. This removes the
            # z/tau scale redundancy that made a raw-logit learnable temperature ill-posed
            # (see issue PAT-244). Bias is dropped (a cosine has no additive offset).
            _rw = router_mod.weight  # [K+1, head_dim]
            _xhat = F.normalize(x_feat.to(torch.float32), dim=-1, eps=1e-6)
            _what = F.normalize(_rw.to(torch.float32), dim=-1, eps=1e-6)
            router_logits = F.linear(_xhat, _what).to(router_logits.dtype)  # cosine in [-1,1]
        if getattr(self, "_pat_g0_cap", None) is not None:  # PAT-243 raw pre-sigmoid router_logits probe (default off)
            self._pat_g0_cap.setdefault("router_logits_raw", []).append(
                (int(lid), router_logits.detach().float().cpu())
            )
        # E2b ablation: replace with globally-learned static logits (not query-conditioned)
        if getattr(self, "wavelet_router_static_learned", False) and self.wavelet_static_router_logits is not None:
            static = self.wavelet_static_router_logits.to(dtype=router_logits.dtype, device=router_logits.device)
            router_logits = static.view(*([1] * (router_logits.dim() - 1)), -1).expand_as(router_logits)
        tau = self._resolve_wavelet_ctxscale_tau(global_step=step_val)
        router_jitter_std_base = getattr(self.config, "router_jitter_flip_ratio", None)
        if router_jitter_std_base is None:
            router_jitter_std_base = getattr(self.config, "router_jitter_std", 0.0)
        router_jitter_max_steps = getattr(self.config, "router_max_steps", 15900)
        router_jitter_std = self._resolve_router_jitter_std(
            router_jitter_std_base,
            global_step=step_val,
            max_steps=router_jitter_max_steps,
        )
        router_jitter_style = "flipratio_token"
        router_jitter_enabled = bool(router_jitter_std > 0.0)
        router_jitter_style_resolved = "flipratio_token"
        router_jitter_target_flip_probability = float("nan")
        default_jitter_stat = float("nan") if router_jitter_enabled else 0.0
        router_jitter_sigma_mean = default_jitter_stat
        router_jitter_sigma_std = default_jitter_stat
        router_jitter_flip_probability_estimate = default_jitter_stat
        if router_jitter_enabled:
            router_logits = self._add_gaussian_jitter(
                router_logits,
                router_jitter_std,
                router_name="ctxscale_router",
                global_step=step_val,
                max_steps=router_jitter_max_steps,
            )
            jitter_stat = getattr(self, "_last_router_jitter_stats", None)
            if isinstance(jitter_stat, dict) and str(jitter_stat.get("router_name", "")) == "ctxscale_router":
                router_jitter_style_resolved = str(jitter_stat.get("style_resolved", router_jitter_style))
                router_jitter_target_flip_probability = float(
                    jitter_stat.get("target_flip_probability", float("nan"))
                )
                router_jitter_sigma_mean = float(jitter_stat.get("sigma_mean", default_jitter_stat))
                router_jitter_sigma_std = float(jitter_stat.get("sigma_std", default_jitter_stat))
                router_jitter_flip_probability_estimate = float(
                    jitter_stat.get("flip_probability_estimate", default_jitter_stat)
                )

        router_jitter_injected = int(router_jitter_enabled and self.training)
        # E2 ablation: fixed uniform router — bypass learned routing weights
        if bool(getattr(self.config, "wavelet_router_fixed_uniform", False)):
            fixed = torch.zeros_like(router_logits)
            fixed[..., 0] = 10.0  # strong non-null gate activation
            router_logits = fixed
        # PAT-225 per-scale knockout: force masked atoms' logits to -1e4 before
        # any routing mode (sigmoid(-1e4)=0, softmax weight -> 0).
        if getattr(self, "wavelet_ctxscale_scale_mask_idx", ()):
            router_logits = router_logits.clone()
            for _mi in self.wavelet_ctxscale_scale_mask_idx:
                if 0 <= int(_mi) < int(self.wavelet_ctxscale_k):
                    router_logits[..., 1 + int(_mi)] = -1e4
        router_sigmoid_mode = str(getattr(self, "wavelet_router_sigmoid_mode", "softmax")).strip().lower()
        if router_sigmoid_mode not in ("softmax", "with_null", "no_null", "with_null_independent_scales", "signed"):
            router_sigmoid_mode = "softmax"
        g0_gate = None
        alpha_gate = None
        sum_g = None
        pi_scale_without_null_gate = None
        nonnull_gate = None
        # PAT-244: optional router-logit normalization / decoupled temperatures.
        # norm_mode="none" (default) leaves behavior byte-identical: tau_null==tau_scale==tau,
        # no RMS, and the with_null / with_null_independent_scales branches below fall through
        # to their original sigmoid+normalize code path.
        _router_norm_mode = str(getattr(self, "wavelet_router_norm_mode", "none")).strip().lower()
        if _router_norm_mode not in (
            "none", "rms_joint", "dual_temp", "dual_temp_scale_rms", "dual_temp_scale_none"
        ):
            _router_norm_mode = "none"
        if _router_norm_mode == "rms_joint":
            # Reproduce the removed pre-2026-07-31 joint RMS over the full [null, scales] vector.
            router_logits = self._rms_norm_last_dim(
                router_logits, eps=float(self.wavelet_ctxscale_router_rms_eps)
            )
        _is_dual_temp = _router_norm_mode in ("dual_temp", "dual_temp_scale_rms", "dual_temp_scale_none")
        if _is_dual_temp:
            # bounded log-sigmoid temperature (see __init__): tau in [tau_min, tau_max]
            _tmin = float(getattr(self, "router_tau_min", 0.1))
            _tmax = float(getattr(self, "router_tau_max", 10.0))
            _tratio = _tmax / _tmin
            tau_null = (_tmin * _tratio ** torch.sigmoid(self.router_tau_null_raw)).to(router_logits.dtype)
            if _router_norm_mode == "dual_temp_scale_none":
                tau_scale = router_logits.new_tensor(1.0)
            else:
                tau_scale = (_tmin * _tratio ** torch.sigmoid(self.router_tau_scale_raw)).to(router_logits.dtype)
        else:
            tau_null = tau
            tau_scale = tau
        # PAT-244: stash effective router temperatures for the per-layer stats line.
        self._last_router_norm_mode = _router_norm_mode
        self._last_router_cosine = int(bool(getattr(self, "wavelet_router_cosine", False)))
        self._last_router_tau_null = float(tau_null.detach()) if torch.is_tensor(tau_null) else float(tau_null)
        self._last_router_tau_scale = float(tau_scale.detach()) if torch.is_tensor(tau_scale) else float(tau_scale)
        if router_sigmoid_mode == "softmax":
            # All options compete together: [null, scale1, ..., scaleK]
            pi = torch.softmax(router_logits / tau, dim=-1)
            pi_scale = pi[..., 1:]
            pi_null = pi[..., 0:1]
            nonnull_gate = (1.0 - pi_null).clamp(min=0.0, max=1.0)
            pi_scale_without_null_gate = pi_scale / nonnull_gate.clamp_min(eps)
            router_mode = "softmax"

        elif router_sigmoid_mode == "with_null":
            # Factorized routing:
            # 1) null vs non-null compete through g0_gate (temperature tau_null)
            # 2) scales compete conditionally inside non-null (temperature tau_scale)
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau_null)          # non-null mass
            # PAT-244: SOFTMAX IS FORBIDDEN here. Keep the original sigmoid-normalize
            # (w = g/sum(g)); dual_temp only (a) applies the decoupled tau_scale and
            # (b) optionally RMS-normalizes the scale logits first. This keeps the
            # normalization FUNCTION identical to the baseline so dual_temp differs
            # from baseline only by the temperature (no softmax-vs-sigmoid confound).
            s_logits = router_logits[..., 1:]
            if _is_dual_temp and _router_norm_mode == "dual_temp_scale_rms":
                s_logits = self._rms_norm_last_dim(
                    s_logits, eps=float(self.wavelet_ctxscale_router_rms_eps)
                )
            g = torch.sigmoid(s_logits / tau_scale)                             # [..., K]
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(eps)
            w = g / sum_g                                                       # conditional scale distribution
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = w
            pi_scale = g0_gate * w                                               # total non-null mass = g0_gate
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_with_null"

        elif router_sigmoid_mode == "no_null":
            # No explicit null logit:
            # mean activation determines total non-null mass
            # scales compete conditionally inside non-null
            g = torch.sigmoid(router_logits[..., 1:] / tau)                      # [..., K]
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(eps)
            w = g / sum_g

            alpha_gate = g.mean(dim=-1, keepdim=True)                            # non-null mass
            nonnull_gate = alpha_gate
            pi_scale_without_null_gate = w
            pi_scale = alpha_gate * w
            pi_null = (1.0 - alpha_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_no_null"

        elif router_sigmoid_mode == "with_null_independent_scales":
            # Factorized routing:
            # 1) null vs non-null compete through g0_gate
            # 2) scales do NOT compete inside non-null; multiple scales can be active together
            #
            # Important:
            # pi here is no longer a probability simplex over [null, scales].
            # pi_null + sum(pi_scale) is generally not 1.
            # This mode should be used only if downstream logic does not require pi to be a normalized distribution.
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau_null)          # non-null gate
            s_logits = router_logits[..., 1:]
            if _is_dual_temp and _router_norm_mode == "dual_temp_scale_rms":
                # PAT-244: RMS-norm ONLY the scale logits (null excluded); scales stay independent.
                s_logits = self._rms_norm_last_dim(
                    s_logits, eps=float(self.wavelet_ctxscale_router_rms_eps)
                )
            g = torch.sigmoid(s_logits / tau_scale)                             # [..., K]
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = g

            pi_scale = g0_gate * g                                               # independent multi-scale activation
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_with_null_independent_scales"

        elif router_sigmoid_mode == "signed":
            # K=1 signed gate: pi_scale = 2*sigmoid(logit)-1 in (-1,1); no null branch
            # (pi=0 is the natural "off"). pi is NOT a probability simplex (like the
            # independent_scales mode). Negative pi flips the ricker's sign at inference.
            g = 2.0 * torch.sigmoid(router_logits[..., 1:] / tau) - 1.0
            nonnull_gate = torch.ones_like(router_logits[..., 0:1])
            pi_scale_without_null_gate = g
            pi_scale = g
            pi_null = torch.zeros_like(router_logits[..., 0:1])
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_signed"

        elif router_sigmoid_mode == "signed_with_null":
            # Signed per-scale weight (can flip sign, unlike with_null_independent_scales'
            # non-negative g), but WITH a shared g0 null/apply gate multiplying every
            # scale -- the signed analog of with_null_independent_scales. Requested to
            # test whether adding g0 back to signed changes its (currently worse-than-K1)
            # result.
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau_null)
            g = 2.0 * torch.sigmoid(router_logits[..., 1:] / tau_scale) - 1.0
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = g
            pi_scale = g0_gate * g
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_signed_with_null"

        elif router_sigmoid_mode == "positive":
            # Non-negative analog of "signed": pi_scale = sigmoid(logit) in (0,1),
            # per scale independently, no shared g0_gate and no null branch (pi=0 is
            # the natural "off" per scale, via gradient pressure alone). Unlike
            # with_null_independent_scales there is no separate g0_gate factor
            # multiplying every scale together -- each scale's weight is a single,
            # unshared quantity, so it can be interpreted directly as that scale's
            # own usefulness (no cross-scale coupling through a shared gate).
            g = torch.sigmoid(router_logits[..., 1:] / tau)
            nonnull_gate = torch.ones_like(router_logits[..., 0:1])
            pi_scale_without_null_gate = g
            pi_scale = g
            pi_null = torch.zeros_like(router_logits[..., 0:1])
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_positive"

        elif router_sigmoid_mode == "gaussian_kernel":
            # Single-decision-maker scale composition: ONE per-query "focus" scalar
            # (not K independent weights) maps through a FIXED, unimodal Gaussian
            # kernel in log2(scale) space to produce the K scale weights, then g0
            # gates overall apply/no-apply exactly as in "with_null". Per-query DOF
            # is 2 (g0 + focus) -- matching K1's DOF=1 plus exactly one new knob,
            # NOT K independent weights like signed/positive. Only router_logits[...,1]
            # (one scalar) is read as the focus logit; router_logits[...,2:] (if K>1)
            # are computed by the router MLP but intentionally unused here, same
            # "computed-but-discarded" pattern as K1's degenerate w or fixedratio's
            # overridden g. sigma is a FIXED hyperparameter (wavelet_ctxscale_kernel_sigma),
            # not learned and not per-query, so it doesn't add a third per-query DOF.
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau_null)
            focus_logit = router_logits[..., 1:2]
            scales_k = self.wavelet_ctxscale_scales.to(
                device=focus_logit.device, dtype=focus_logit.dtype
            )
            log_scales = torch.log2(scales_k)
            log_min = log_scales.min()
            log_max = log_scales.max()
            mu = torch.sigmoid(focus_logit) * (log_max - log_min) + log_min  # [...,1]
            sigma = float(self.wavelet_ctxscale_kernel_sigma)
            log_scales_b = log_scales.view(*([1] * (mu.dim() - 1)), -1)  # [1,...,1,K]
            g = torch.exp(-(log_scales_b - mu).pow(2) / (2.0 * sigma * sigma))  # [...,K]
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = g
            pi_scale = g0_gate * g
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            router_mode = "sigmoid_gaussian_kernel"

        else:
            raise ValueError(f"Unknown router_sigmoid_mode: {router_sigmoid_mode}")

        if bool(getattr(self, "wavelet_ctxscale_unconditional_rms", False)) and not use_mlp_bias_baseline:
            # PAT-244: unconditional RMS wavelet -- override whatever the router branch
            # above computed. No selection (K must be 1, enforced below), no null gate:
            # the single scale's weight is a constant 1.0, so the (still RMS-normalized,
            # still shift-applied) basis is added to the logits unconditionally. The
            # router module ran above (router_logits computed) but its output is
            # discarded here, so wavelet_ctx_router receives zero gradient in this mode.
            if K != 1:
                raise ValueError(
                    "wavelet_ctxscale_unconditional_rms requires wavelet_ctxscale_k==1 "
                    f"(no scale selection is defined for K>1); got K={K}."
                )
            pi_null = torch.zeros_like(router_logits[..., 0:1])
            pi_scale = torch.ones_like(router_logits[..., 1:2])
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            g0_gate = torch.ones_like(pi_null)
            nonnull_gate = torch.ones_like(pi_null)
            pi_scale_without_null_gate = pi_scale
            router_mode = "unconditional_rms"

        if getattr(self, "wavelet_ctxscale_fixed_scale_ratio_buf", None) is not None and not use_mlp_bias_baseline:
            # Keep g0 learnable/query-conditioned exactly as with_null_independent_scales,
            # but replace the learned per-scale mixture with a fixed constant ratio.
            if router_sigmoid_mode != "with_null_independent_scales":
                raise ValueError(
                    "wavelet_ctxscale_fixed_scale_ratio requires "
                    f"wavelet_router_sigmoid_mode='with_null_independent_scales', got {router_sigmoid_mode!r}."
                )
            fixed_ratio = self.wavelet_ctxscale_fixed_scale_ratio_buf.to(dtype=g0_gate.dtype, device=g0_gate.device)
            fixed_ratio = fixed_ratio.view(*([1] * (g0_gate.dim() - 1)), fixed_ratio.shape[-1])
            pi_scale = g0_gate * fixed_ratio
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = fixed_ratio.expand_as(pi_scale)
            router_mode = "sigmoid_with_null_fixed_scale_ratio"

        if getattr(self, "wavelet_ctxscale_ratio_learnable", False) and not use_mlp_bias_baseline:
            # Query-INDEPENDENT but LEARNED per-scale ratio: same structural role as the
            # fixed_scale_ratio override above (g0 stays learnable/query-conditioned,
            # only the scale-mixture is replaced), except the mixture is a single
            # nn.Parameter optimized by gradient descent rather than a hand-set constant.
            if router_sigmoid_mode != "with_null_independent_scales":
                raise ValueError(
                    "wavelet_ctxscale_ratio_learnable requires "
                    f"wavelet_router_sigmoid_mode='with_null_independent_scales', got {router_sigmoid_mode!r}."
                )
            learned_ratio = torch.sigmoid(self.wavelet_ctxscale_ratio_param).to(
                dtype=g0_gate.dtype, device=g0_gate.device
            )
            learned_ratio = learned_ratio.view(*([1] * (g0_gate.dim() - 1)), learned_ratio.shape[-1])
            pi_scale = g0_gate * learned_ratio
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            nonnull_gate = g0_gate
            pi_scale_without_null_gate = learned_ratio.expand_as(pi_scale)
            router_mode = "sigmoid_with_null_learnable_static_ratio"

        if getattr(self, "wavelet_ctxscale_g0_learnable", False) and not use_mlp_bias_baseline:
            # PAT-253: query-INDEPENDENT but LEARNED null/apply gate. Runs after every
            # scale-mixture branch above so it composes with whichever mixture is
            # currently in effect (default per-query g, fixed_scale_ratio, or
            # ratio_learnable) via pi_scale_without_null_gate, which each of those
            # branches already sets -- this override only replaces g0_gate itself.
            if router_sigmoid_mode not in ("with_null_independent_scales", "with_null"):
                raise ValueError(
                    "wavelet_ctxscale_g0_learnable requires wavelet_router_sigmoid_mode "
                    f"in ('with_null_independent_scales', 'with_null'), got {router_sigmoid_mode!r}."
                )
            g0_gate = torch.sigmoid(self.wavelet_ctxscale_g0_param).to(
                dtype=router_logits.dtype, device=router_logits.device
            )
            g0_gate = g0_gate.view(*([1] * (router_logits.dim() - 1)), 1) * torch.ones_like(
                router_logits[..., 0:1]
            )
            pi_scale = g0_gate * pi_scale_without_null_gate
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            nonnull_gate = g0_gate
            router_mode = router_mode + "_g0static"

        if getattr(self, "wavelet_ctxscale_g0_fixed_value", None) is not None and not use_mlp_bias_baseline:
            # PAT-253: hand-set constant null/apply gate, zero learnable parameters
            # for this decision. Same composition pattern as g0_learnable above --
            # reuses whatever pi_scale_without_null_gate the mixture branch set --
            # except g0_gate is a literal constant, not even an nn.Parameter.
            if router_sigmoid_mode not in ("with_null_independent_scales", "with_null"):
                raise ValueError(
                    "wavelet_ctxscale_g0_fixed_value requires wavelet_router_sigmoid_mode "
                    f"in ('with_null_independent_scales', 'with_null'), got {router_sigmoid_mode!r}."
                )
            g0_gate = torch.full_like(
                router_logits[..., 0:1], float(self.wavelet_ctxscale_g0_fixed_value)
            )
            pi_scale = g0_gate * pi_scale_without_null_gate
            pi_null = (1.0 - g0_gate).clamp(min=0.0, max=1.0)
            pi = torch.cat([pi_null, pi_scale], dim=-1)
            nonnull_gate = g0_gate
            router_mode = router_mode + "_g0fixed"

        # Apply layer/query knockout to the final router weights. Doing this to
        # raw router logits is incorrect for rms_joint because normalization
        # maps a sentinel such as -1e4 back to a finite value.
        _ko_this_layer = (
            int(self.layer_idx or 0)
            in getattr(self, "wavelet_ctxscale_ko_layers_idx", ())
        )
        if _ko_this_layer:
            pi = pi.clone()
            _ranges = getattr(self, "wavelet_ctxscale_ko_query_ranges_idx", ())
            if not _ranges:
                pi[..., 0] = 1.0
                pi[..., 1:] = 0.0
                if g0_gate is not None:
                    g0_gate = torch.zeros_like(g0_gate)
                nonnull_gate = torch.zeros_like(nonnull_gate)
            else:
                _T_router = int(pi.shape[-2])
                if g0_gate is not None:
                    g0_gate = g0_gate.clone()
                nonnull_gate = nonnull_gate.clone()
                for _start, _end in _ranges:
                    _start = min(int(_start), _T_router)
                    _end = min(int(_end), _T_router)
                    if _start >= _end:
                        continue
                    pi[..., _start:_end, 0] = 1.0
                    pi[..., _start:_end, 1:] = 0.0
                    if g0_gate is not None:
                        g0_gate[..., _start:_end, :] = 0.0
                    nonnull_gate[..., _start:_end, :] = 0.0

        if getattr(self, "_pat_g0_cap", None) is not None:  # PAT-243 g0_gate-by-position probe (default off)
            _g0_for_cap = g0_gate if g0_gate is not None else nonnull_gate
            self._pat_g0_cap.setdefault("g0_gate", []).append(
                (int(lid), _g0_for_cap.detach().float().cpu())
            )

        pi = self._apply_ctxscale_do_intervention(
            pi=pi,
            layer_idx=lid,
        )
        # IMPORTANT: pi_scale must be derived from post-intervention pi.
        # Otherwise do(scale) has no effect on the final bias path.
        pi_scale = pi[..., 1:]
        if pi.dim() != 3:
            raise ValueError(
                "Head-wise wavelet routing has been removed; "
                f"router probabilities must be 3D, got {tuple(pi.shape)}."
            )
        # PAT-244: expand from K_distinct (one router decision per distinct
        # scale in wavelet_ctxscale_scale_max_exp) to K_total = K_distinct *
        # shift_number (one wavelet slot per independently-learned-shift copy)
        # via repeat_interleave -- same consecutive-block ordering as the
        # scales buffer built in __init__, so slot i's router weight always
        # matches slot i's scale/shift. do(scale) intervention above still
        # operates in K_distinct-space (unchanged semantics for existing
        # do-intervention callers); every copy of an intervened-on scale
        # inherits the same forced weight through this expansion.
        _shift_number = int(getattr(self, "wavelet_ctxscale_shift_number", 1))
        if _shift_number > 1:
            pi_scale = pi_scale.repeat_interleave(_shift_number, dim=-1)
            pi = torch.cat([pi[..., 0:1], pi_scale], dim=-1)
            if pi_scale_without_null_gate is not None:
                pi_scale_without_null_gate = pi_scale_without_null_gate.repeat_interleave(_shift_number, dim=-1)
        # Router usage capture (analysis-only, no effect on the bias path):
        # pi[..., 0] is the null gate, pi[..., 1:] are the K final per-scale
        # weights actually used below to build the bias -- this is the
        # "which scale gets used" statistic for K>1 usage plots. QWAB-only
        # (PA-only checkpoints never reach this function).
        self._last_router_pi = pi.detach().to(torch.float32)
        router_mode_is_signed = router_mode == "sigmoid_signed"
        # "rms_both": per-scale RMS-normalize each basis (like "none"/"sqrt"/"k"),
        # THEN weighted-sum, THEN ALSO apply the outer joint RMS over the sum (like
        # "rms"). Isolates cross-scale raw-amplitude fairness (different rho values
        # have different raw RMS over a fixed causal window -- see PAT-225 comment)
        # from the "rms" mode's separate auto-gain/shrink-to-zero issue, which this
        # mode does NOT fix (outer RMS still reinflates a near-zero weighted sum).
        skip_per_scale_basis_norm = (
            self.multiscale_norm_requested == "rms"
            and int(self.wavelet_ctxscale_k_total) > 1
        )
        multiscale_rms_after_sum = (
            self.multiscale_norm_requested in ("rms", "rms_both")
            and int(self.wavelet_ctxscale_k_total) > 1
        )

        def _router_diag_prob_dist(pi_tensor: torch.Tensor) -> torch.Tensor:
            if router_mode_is_signed:
                pi_prob = pi_tensor.abs()
                return pi_prob / pi_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return pi_tensor.clamp_min(1e-12)

        # Entropy regularization (training only).
        # floor mode (default): penalize entropy below floor — keeps routing diverse.
        # ceiling mode (router_entropy_ceiling > 0): penalize entropy above ceiling — encourages selective routing.
        router_entropy_reg_enable = bool(getattr(self.config, "router_entropy_reg_enable", False))
        router_entropy_reg_lambda = float(getattr(self.config, "router_entropy_reg_lambda", 0.0))
        router_entropy_floor = float(getattr(self.config, "router_entropy_floor", 1.72))
        router_entropy_ceiling = float(getattr(self.config, "router_entropy_ceiling", -1.0))
        router_entropy_reg_loss = reg_zero
        router_entropy_reg_active_frac = reg_zero.detach()
        if self.training and router_entropy_reg_enable and router_entropy_reg_lambda > 0.0:
            pi_reg = _router_diag_prob_dist(pi.to(device=device, dtype=torch.float32))
            pi_entropy_reg = -(pi_reg * (pi_reg + 1e-8).log()).sum(dim=-1)
            if router_entropy_ceiling > 0.0:
                reg_penalty = F.relu(pi_entropy_reg - router_entropy_ceiling)
            else:
                reg_penalty = F.relu(router_entropy_floor - pi_entropy_reg)
            router_entropy_reg_loss = reg_penalty.mean() * router_entropy_reg_lambda
            router_entropy_reg_active_frac = (reg_penalty > 0).to(torch.float32).mean().detach()
        self._last_router_entropy_reg_loss = router_entropy_reg_loss
        self._last_router_entropy_reg_active_frac = router_entropy_reg_active_frac

        do_stat = getattr(self, "_last_ctxscale_do_stat", None)
        do_active = bool(
            isinstance(do_stat, dict)
            and bool(do_stat.get("enabled", False))
            and int(do_stat.get("layer", -1)) == int(lid)
        )
        do_mode = str(do_stat.get("mode", "")) if do_active else ""
        do_scope = str(do_stat.get("scope", "none")) if do_active else "none"
        do_target_heads = []
        do_target_head_for_check = -1
        if do_active:
            try:
                do_target_heads = [int(x) for x in list(do_stat.get("target_heads", []))]
            except Exception:
                do_target_heads = []
            do_target_head_for_check = int(do_stat.get("target_head_for_check", -1))
            if multiscale_rms_after_sum:
                # Intervention edits pi directly, so rebuild the separated
                # scale-mix and non-null gate from the post-intervention pi.
                nonnull_gate = (1.0 - pi[..., 0:1]).clamp(min=0.0, max=1.0)
                pi_scale_without_null_gate = pi_scale / nonnull_gate.clamp_min(eps)
            if do_target_head_for_check >= 0 and do_target_head_for_check not in do_target_heads:
                do_target_heads = [do_target_head_for_check] + do_target_heads
            H_for_do = int(E_base_raw.shape[1])
            do_target_heads = sorted(set(h for h in do_target_heads if 0 <= int(h) < H_for_do))

        # Save scale-only probabilities (exclude null channel) for case-level scale analysis.
        if K > 0:
            pi_scale_full = pi[..., 1 : (K + 1)]
            if pi_scale_full.dim() == 3:
                # [B,T,K] -> expand to [B,T,H,K] for head-wise tooling compatibility.
                pi_scale_full = pi_scale_full.unsqueeze(2).expand(-1, -1, int(self.num_heads), -1)
            elif pi_scale_full.dim() == 4:
                pass
            else:
                pi_scale_full = None
            if pi_scale_full is not None:
                self._last_ctxscale_router_prob = pi_scale_full.detach().to(dtype=torch.float32)
        non_null_mass = None
        if g0_gate is not None:
            non_null_mass = g0_gate
        elif alpha_gate is not None:
            non_null_mass = alpha_gate
        elif torch.is_tensor(pi_scale):
            if router_mode_is_signed:
                non_null_mass = pi_scale.abs().sum(dim=-1, keepdim=True)
            else:
                non_null_mass = pi_scale.sum(dim=-1, keepdim=True)
        if torch.is_tensor(non_null_mass):
            self._last_ctxscale_non_null_mass = non_null_mass.detach().to(dtype=torch.float32)
        if torch.is_tensor(pi_null):
            self._last_ctxscale_null_mass = pi_null.detach().to(dtype=torch.float32)

        shift_ln = self.mlp_bias_shift_ln if use_mlp_bias_baseline else self.wavelet_shift_ln
        shift_proj = self.mlp_bias_shift_proj if use_mlp_bias_baseline else self.wavelet_shift_proj
        # PAT-244: reuse wavelet_ctx_feat_detach_delta (the router feature's detach
        # switch) for the shift path too, instead of a separate flag -- same
        # gradient-isolation intent (this module's own weights still train, but
        # don't reshape the backbone), one knob controls both consistently.
        _shift_input = hidden_states
        if bool(getattr(self, "wavelet_ctx_feat_detach_delta", False)) and not use_mlp_bias_baseline:
            _shift_input = _shift_input.detach()
        h_ln = shift_ln(_shift_input.to(device=device, dtype=shift_ln.weight.dtype))
        shift_per_scale = bool(getattr(self, "wavelet_ctxscale_shift_per_scale", False)) and not use_mlp_bias_baseline
        if shift_per_scale:
            # [B, T, K] -- one independently-learned shift decision per scale index,
            # instead of one shared decision broadcast to every scale.
            rho = torch.sigmoid(shift_proj(h_ln))
        else:
            rho = torch.sigmoid(shift_proj(h_ln).squeeze(-1))
        # Causal rho intervention: override rho to a fixed constant for ablation studies.
        # Set wavelet_ctxscale_rho_override=<float in [0,1]> in supply_model.cfg to activate.
        _rho_override = getattr(self, "wavelet_ctxscale_rho_override", None)
        if _rho_override is not None:
            rho = torch.full_like(rho, float(_rho_override))
        use_scale_coupled_shift = bool(getattr(self, "wavelet_ctxscale_scale_dependent_shift", False))
        use_abs_shift_causal = bool(getattr(self, "wavelet_ctxscale_abs_shift_causal", False))
        shift_t_mode = str(getattr(self, "wavelet_shift_T_mode", "legacy")).strip().lower()
        if shift_t_mode not in ("legacy", "runtime", "train_ref"):
            shift_t_mode = "legacy"
        T_test = int(T)
        if shift_t_mode == "train_ref":
            T_used = max(2, int(getattr(self, "wavelet_shift_T_ref", T_test)))
        else:
            T_used = max(2, int(T_test))
        apply_shift_t_scaling = shift_t_mode in ("runtime", "train_ref")
        shift_t_scale = float(max(1, T_used - 1))
        if use_abs_shift_causal and apply_shift_t_scaling:
            raise ValueError(
                "abs_shift_causal ignores T_used; disable abs_shift_causal or set wavelet_shift_T_mode=legacy."
            )
        if use_scale_coupled_shift:
            # Shift is defined in scale units and later coupled with each wavelet scale.
            shift_unit_max = float(getattr(self, "wavelet_ctxscale_shift_unit_max", 1.0))
            beta_m = (2.0 * rho - 1.0) * shift_unit_max
        else:
            if use_abs_shift_causal:
                # Causal absolute-position shift: each query q selects center from [0, q].
                q_pos = torch.arange(T_test, device=device, dtype=torch.float32).view(1, T_test, 1) if shift_per_scale else torch.arange(T_test, device=device, dtype=torch.float32).view(1, T_test)
                beta_m = torch.round(rho * q_pos)
                beta_m = torch.minimum(beta_m, q_pos).clamp_min_(0.0)
            else:
                # Legacy token-index shift.
                beta_upper = max(1, int(T_used - 1)) if apply_shift_t_scaling else max(1, int(T_test - 1))
                if bool(getattr(self, "wavelet_ctxscale_shift_legacy_symmetric", False)):
                    # PAT-244: opt-in symmetric variant, range [-beta_upper, +beta_upper] instead of
                    # the default one-directional [0, beta_upper]. Combine with wavelet_shift_T_mode=
                    # train_ref + wavelet_shift_T_ref=L_train to keep beta_upper fixed to training
                    # length regardless of L_test, instead of the legacy default (T_test-1, which
                    # implicitly grows the shift range at longer test lengths).
                    beta_m = (2.0 * rho - 1.0) * float(beta_upper)
                else:
                    beta_m = torch.round(rho * float(beta_upper)).clamp_(0.0, float(beta_upper))

        self._last_ctxscale_rho = rho.detach().to(dtype=torch.float32)
        self._last_ctxscale_beta_m = beta_m.detach().to(dtype=torch.float32)
        self._last_ctxscale_shift_unit_max = float(getattr(self, "wavelet_ctxscale_shift_unit_max", 1.0))
        self._last_ctxscale_use_scale_coupled_shift = bool(use_scale_coupled_shift)

        if self.wavelet_logit_bias_debug_assert:
            if not torch.isfinite(pi).all():
                raise FloatingPointError("ctxscale_shift_v0: non-finite router pi")
            if not torch.isfinite(rho).all():
                raise FloatingPointError("ctxscale_shift_v0: non-finite rho")
            if not torch.isfinite(beta_m).all():
                raise FloatingPointError("ctxscale_shift_v0: non-finite beta")
            if not use_scale_coupled_shift:
                if use_abs_shift_causal:
                    q_pos = torch.arange(T_test, device=device, dtype=torch.float32).view(1, T_test)
                    if float((beta_m < 0.0).any().item()) != 0.0:
                        raise FloatingPointError("ctxscale_shift_v0: beta_token below 0 in abs causal mode")
                    if float((beta_m > q_pos).any().item()) != 0.0:
                        raise FloatingPointError("ctxscale_shift_v0: beta_token above q in abs causal mode")
                else:
                    beta_upper = float(max(1, int(T_used - 1)) if apply_shift_t_scaling else max(1, int(T_test - 1)))
                    if float(beta_m.detach().amin().item()) < 0.0 or float(beta_m.detach().amax().item()) > beta_upper:
                        raise FloatingPointError("ctxscale_shift_v0: beta_token out of range [0, T_used-1]")

        clamp_abs = float(self.wavelet_ctxscale_lock_clamp_abs)
        g_max = float(getattr(self.config, "wavelet_ctxscale_g_max", self.wavelet_ctxscale_g_max))
        gate_param = self.mlp_bias_logit_bias_a if use_mlp_bias_baseline else self.wavelet_logit_bias_a
        gate_head_param = self.mlp_bias_logit_bias_a_head if use_mlp_bias_baseline else self.wavelet_logit_bias_a_head
        # g_layer removed (2026-08-01): the learnable layer/head gate was a source of
        # silent inconsistency across PAT-234 configs (K1 configs had it disabled,
        # K2-K5 independent_rms configs had it active). Force g_layer=1.0 unconditionally
        # so wavelet_ctxscale_disable_layer_gate/wavelet_ctxscale_use_head_gate no longer
        # have any effect, regardless of what a supply_model.cfg requests.
        use_head_gate = False
        disable_layer_gate = True
        gate_branch = "layer_gate_active"
        if use_head_gate:
            gate_branch = "head_gate_active"
        elif disable_layer_gate:
            gate_branch = "layer_gate_disabled"
        # One-time per-layer branch/config log for both train/eval debugging.
        # This makes it explicit whether "disable_layer_gate" is actually taking effect.
        gate_log_seen = getattr(self, "_ctxscale_gate_branch_log_seen", None)
        if gate_log_seen is None:
            gate_log_seen = set()
            self._ctxscale_gate_branch_log_seen = gate_log_seen
        gate_log_key = (
            int(lid),
            str(wavelet_mode_resolved),
            str(gate_branch),
            int(bool(getattr(self, "wavelet_ctxscale_disable_layer_gate", False))),
            int(bool(getattr(self, "wavelet_ctxscale_use_head_gate", False))),
        )
        if gate_log_key not in gate_log_seen:
            self._k1_emit_log(
                f"[path_attn gate check] layer={int(lid)} step={int(step_val)} "
                f"wavelet_mode={wavelet_mode_resolved} gate_branch={gate_branch} "
                f"cfg_disable_layer_gate={int(bool(getattr(self, 'wavelet_ctxscale_disable_layer_gate', False)))} "
                f"cfg_use_head_gate={int(bool(getattr(self, 'wavelet_ctxscale_use_head_gate', False)))} "
                f"param_has_head_gate={int(gate_head_param is not None)}"
            )
            gate_log_seen.add(gate_log_key)
        g_head = None
        if use_head_gate:
            g_head_raw = gate_head_param.to(device=device, dtype=torch.float32)
            g_head_raw_clipped = g_head_raw.clamp(min=-clamp_abs, max=clamp_abs)
            g_head_raw_used = g_head_raw + (g_head_raw_clipped - g_head_raw).detach()
            g_head_raw_clamped = int(bool((g_head_raw.detach().abs() > clamp_abs).any().item()))
            g_head = g_max * torch.sigmoid(g_head_raw_used)
            if not torch.isfinite(g_head).all():
                _warn_nonfinite("g_head")
                g_head = torch.nan_to_num(
                    g_head,
                    nan=0.0,
                    posinf=g_max,
                    neginf=0.0,
                )
            g_layer = g_head.mean()
            g_layer_raw = g_head_raw.mean()
            g_layer_raw_used = g_head_raw_used.mean()
            sat_low_frac = float((g_head < 0.01 * float(g_max)).float().mean().item())
            sat_high_frac = float((g_head > 0.99 * float(g_max)).float().mean().item())
            gate_state = {
                "sig": float((g_layer / max(float(g_max), 1e-12)).item()),
                "sat_low": int(sat_low_frac > 0.7),
                "sat_high": int(sat_high_frac > 0.7),
                "sat_extreme": int((sat_low_frac > 0.7) or (sat_high_frac > 0.7)),
                "grad_abs": float("nan"),
                "grad_abs_p50": float("nan"),
                "grad_abs_p90": float("nan"),
                "grad_abs_max": float("nan"),
                "grad_zero_ratio": float("nan"),
                "grad_finite_ratio": float("nan"),
                "grad_nonfinite": -1,
                "grad_missing": 0,
                "grad_zero": 0,
                "delta_a": float("nan"),
                "update_ratio": float("nan"),
                "locked": 0,
            }
            autofix_active = 0
            g_layer_raw_clamped = int(g_head_raw_clamped)
        elif disable_layer_gate:
            # Ablation: remove learnable layer-wise gate and keep full bias strength.
            g_layer_raw = gate_param.to(device=device, dtype=torch.float32)
            g_layer_raw_used = g_layer_raw
            g_layer_raw_clamped = 0
            g_layer = torch.ones((), device=device, dtype=torch.float32)
            gate_state = {
                "sig": 1.0,
                "sat_low": 0,
                "sat_high": 0,
                "sat_extreme": 0,
                "grad_abs": float("nan"),
                "grad_abs_p50": float("nan"),
                "grad_abs_p90": float("nan"),
                "grad_abs_max": float("nan"),
                "grad_zero_ratio": float("nan"),
                "grad_finite_ratio": float("nan"),
                "grad_nonfinite": -1,
                "grad_missing": 0,
                "grad_zero": 0,
                "delta_a": float("nan"),
                "update_ratio": float("nan"),
                "locked": 0,
            }
            autofix_active = 0
        else:
            g_layer_raw = gate_param.to(device=device, dtype=torch.float32)
            gate_state = self._ctxscale_gate_state(step=step_val, g_max=g_max, g_layer_raw=g_layer_raw)
            if (
                need_log
                and self.training
                and torch.is_grad_enabled()
                and bool(getattr(self, "_wavelet_gate_grad_seen", False))
                and int(gate_state.get("grad_nonfinite", 0)) > 0
            ):
                _warn_nonfinite("gate_grad")
            autofix_active = int(bool(self.wavelet_gate_autofix) and int(gate_state.get("locked", 0)) == 1)
            if autofix_active == 1:
                clamp_abs = min(clamp_abs, float(self.wavelet_gate_autofix_clamp_abs))
            # Forward clamp for numeric safety, but keep identity gradient (STE-style)
            # so gates initialized outside clamp range can still be optimized back.
            g_layer_raw_clipped = g_layer_raw.clamp(min=-clamp_abs, max=clamp_abs)
            g_layer_raw_used = g_layer_raw + (g_layer_raw_clipped - g_layer_raw).detach()
            g_layer_raw_clamped = int(bool((g_layer_raw.detach().abs() > clamp_abs).item()))
            g_layer = g_max * torch.sigmoid(g_layer_raw_used.to(device=device, dtype=torch.float32))
            if not torch.isfinite(g_layer).all():
                _warn_nonfinite("g_layer")
                g_layer = torch.nan_to_num(
                    g_layer,
                    nan=0.0,
                    posinf=g_max,
                    neginf=0.0,
                )
        logits_out = E_base_raw.to(dtype=torch.float32).clone()
        self._last_logits_pa_only = E_base_raw.detach().to(dtype=torch.float32)
        _pa_dampen_threshold = int(getattr(self, "wavelet_pa_beyond_dampen_threshold", 0))
        _pa_dampen_alpha = float(getattr(self, "wavelet_pa_beyond_dampen_alpha", 1.0))
        if _pa_dampen_threshold > 0 and _pa_dampen_alpha != 1.0:
            _q_pos = torch.arange(logits_out.shape[-2], device=logits_out.device, dtype=torch.long).view(1, 1, -1, 1)
            _k_pos = torch.arange(logits_out.shape[-1], device=logits_out.device, dtype=torch.long).view(1, 1, 1, -1)
            _beyond_mask = (_q_pos - _k_pos) >= _pa_dampen_threshold
            logits_out = torch.where(_beyond_mask, logits_out * _pa_dampen_alpha, logits_out)
        _pa_within_threshold = int(getattr(self, "wavelet_pa_within_dampen_threshold", 0))
        _pa_within_alpha = float(getattr(self, "wavelet_pa_within_dampen_alpha", 1.0))
        if _pa_within_threshold > 0 and _pa_within_alpha != 1.0:
            _q_pos = torch.arange(logits_out.shape[-2], device=logits_out.device, dtype=torch.long).view(1, 1, -1, 1)
            _k_pos = torch.arange(logits_out.shape[-1], device=logits_out.device, dtype=torch.long).view(1, 1, 1, -1)
            _dist = _q_pos - _k_pos
            _within_mask = (_dist >= 0) & (_dist < _pa_within_threshold)
            logits_out = torch.where(_within_mask, logits_out * _pa_within_alpha, logits_out)
        if self.wavelet_logit_bias_debug_assert:
            assert E_base_raw.dim() == 4

        diff = torch.arange(T, device=device, dtype=torch.float32)
        # E3 ablation: shift wavelet position reference by a learnable global scalar
        if getattr(self, "wavelet_ctxscale_learnable_anchor", False) and self.wavelet_anchor_offset is not None:
            diff = diff + self.wavelet_anchor_offset.to(dtype=torch.float32, device=device)
        q_chunk = max(1, min(int(self.wavelet_ctxscale_chunk_q), T))
        far_only = bool(self.wavelet_ctxscale_far_only) and int(self.wavelet_ctxscale_far_min_delta) > 0
        far_over_delta = int(getattr(self, "wavelet_ctxscale_far_over_delta", 0))
        far_over_alpha = float(getattr(self, "wavelet_ctxscale_far_over_alpha", 1.0))
        apply_far_over = (far_over_delta > 0) and (far_over_alpha < 0.999999)
        k_pos_long = torch.arange(T, device=device, dtype=torch.long).view(1, 1, T) if (far_only or apply_far_over) else None
        head_mask = None
        valid_heads = None
        if self.wavelet_ctxscale_head_indices is not None:
            h_total = int(E_base_raw.shape[1])
            head_mask = torch.zeros((1, h_total, 1, 1), device=device, dtype=torch.float32)
            valid_heads = [h for h in self.wavelet_ctxscale_head_indices if 0 <= int(h) < h_total]
            if len(valid_heads) > 0:
                head_mask[:, valid_heads, :, :] = 1.0

        do_target_head = int(do_target_head_for_check if do_target_head_for_check >= 0 else -1)
        if do_target_head < 0 and len(do_target_heads) > 0:
            do_target_head = int(do_target_heads[0])
        do_non_target_heads_sample = []
        if do_active:
            H_full = int(E_base_raw.shape[1])
            do_non_target_heads_sample = [h for h in range(H_full) if h not in set(do_target_heads)][:2]
        do_target_sum_sq = 0.0
        do_target_sum = 0.0
        do_target_count = 0
        do_target_maxabs = 0.0
        do_non_target_acc = {
            int(h): {"sum_sq": 0.0, "sum": 0.0, "count": 0, "maxabs": 0.0}
            for h in do_non_target_heads_sample
        }

        sample_bias_vals = []
        sample_eff_vals = []
        sample_base_vals = []
        sample_scale_vals = []
        sample_shift_vals = []
        sample_sraw_vals = []
        sample_traw_vals = []
        norm_scale_vals = []
        norm_keff_vals = []
        norm_pre_sum_sq = torch.zeros((), device=device, dtype=torch.float32)
        norm_post_sum_sq = torch.zeros((), device=device, dtype=torch.float32)
        norm_elem_count = 0
        film_nf_flags = {"s_raw": 0, "t_raw": 0, "scale": 0, "shift": 0}
        sat_s_num = 0
        sat_t_num = 0
        sat_den = 0
        sample_budget = max(128, int(self.wavelet_ctxscale_max_log_samples))
        sample_count = 0
        if need_log:
            t_take = max(1, min(int(self.wavelet_logit_bias_log_sample_tokens), T))
            sample_q_idx = torch.linspace(0, T - 1, steps=t_take, device=device).long()
            sample_k_idx = sample_q_idx
        else:
            sample_q_idx = None
            sample_k_idx = None

        analysis_enabled = self._wavelet_analysis_enabled_for_layer(layer_idx=lid, need_log=bool(need_log))
        analysis_sample_q_idx = None
        analysis_energy_sum = None
        analysis_abs_sum = None
        analysis_count = None
        analysis_width_vals = None
        analysis_eff_abs_vals = None
        analysis_q_count = 0
        analysis_qk_count = 0
        if self._wavelet_export_kind() == "viz":
            analysis_q_cap = max(1, int(getattr(self, "wavelet_viz_sample_q", 64)))
            analysis_k_cap = max(1, int(getattr(self, "wavelet_viz_sample_k", 256)))
        else:
            analysis_q_cap = max(1, int(getattr(self, "wavelet_analysis_max_q", 64)))
            analysis_k_cap = 256
        if analysis_enabled:
            if sample_q_idx is not None:
                if int(sample_q_idx.numel()) > int(analysis_q_cap):
                    a_take = max(1, int(analysis_q_cap))
                    a_pick = torch.linspace(
                        0,
                        int(sample_q_idx.numel()) - 1,
                        steps=a_take,
                        device=device,
                    ).long()
                    analysis_sample_q_idx = sample_q_idx.index_select(0, a_pick)
                else:
                    analysis_sample_q_idx = sample_q_idx
            else:
                t_take_a = max(
                    1,
                    min(
                        int(analysis_q_cap),
                        T,
                    ),
                )
                analysis_sample_q_idx = torch.linspace(0, T - 1, steps=t_take_a, device=device).long()
            analysis_energy_sum = [0.0 for _ in range(K)]
            analysis_abs_sum = [0.0 for _ in range(K)]
            analysis_count = [0.0 for _ in range(K)]
            analysis_width_vals = [[] for _ in range(K)]
            analysis_eff_abs_vals = []

        def _analysis_k_idx_for_q(q_abs_val: int) -> Optional[torch.Tensor]:
            if q_abs_val < 0:
                return None
            k_take = min(int(analysis_k_cap), int(q_abs_val) + 1, int(T))
            if k_take <= 0:
                return None
            if k_take >= (int(q_abs_val) + 1):
                return torch.arange(0, int(q_abs_val) + 1, device=device, dtype=torch.long)
            k_idx = torch.linspace(0.0, float(q_abs_val), steps=int(k_take), device=device).long()
            if int(k_idx.numel()) == 0:
                return None
            return torch.unique(k_idx, sorted=True)

        def _analysis_accumulate_component(
            stat_idx: int,
            contrib_full: torch.Tensor,
            beta_local: torch.Tensor,
            q_local: Optional[torch.Tensor],
            q_abs: Optional[torch.Tensor],
        ):
            nonlocal analysis_qk_count
            if (
                not analysis_enabled
                or q_local is None
                or q_abs is None
                or stat_idx < 0
                or stat_idx >= K
                or contrib_full is None
                or beta_local is None
                or q_local.numel() == 0
            ):
                return
            for j in range(int(q_local.numel())):
                q_rel = int(q_local[j].item())
                q_abs_val = int(q_abs[j].item())
                k_idx = _analysis_k_idx_for_q(q_abs_val)
                if k_idx is None or int(k_idx.numel()) == 0:
                    continue
                c_q = contrib_full[:, q_rel, :].index_select(-1, k_idx)
                abs_c = c_q.abs()
                analysis_energy_sum[stat_idx] += float(c_q.square().sum().item())
                analysis_abs_sum[stat_idx] += float(abs_c.sum().item())
                c_num = float(B * int(k_idx.numel()))
                analysis_count[stat_idx] += c_num
                if stat_idx == 0:
                    analysis_qk_count += int(c_num)
                beta_q = beta_local[:, q_rel]
                dist2 = (k_idx.to(dtype=torch.float32).view(1, -1) - beta_q.unsqueeze(-1)).pow(2)
                den = abs_c.sum(dim=-1).clamp_min(1e-12)
                width = torch.sqrt((dist2 * abs_c).sum(dim=-1) / den)
                analysis_width_vals[stat_idx].extend(width.detach().reshape(-1).cpu().tolist())

        for q0 in range(0, T, q_chunk):
            q1 = min(T, q0 + q_chunk)
            q_len = int(q1 - q0)
            far_mask = None
            far_over_mask = None
            if far_only or apply_far_over:
                q_pos_long = torch.arange(q0, q1, device=device, dtype=torch.long).view(1, q_len, 1)
                delta_long = q_pos_long - k_pos_long
            if far_only:
                far_mask = (delta_long >= int(self.wavelet_ctxscale_far_min_delta)) & (delta_long >= 0)
            if apply_far_over:
                far_over_mask = (delta_long > int(far_over_delta)) & (delta_long >= 0)

            analysis_q_local = None
            analysis_q_abs = None
            if analysis_enabled and analysis_sample_q_idx is not None:
                local_a = (analysis_sample_q_idx >= q0) & (analysis_sample_q_idx < q1)
                if bool(local_a.any()):
                    q_abs_a = analysis_sample_q_idx[local_a]
                    analysis_q_local = q_abs_a - q0
                    analysis_q_abs = q_abs_a
                    analysis_q_count += int(B * int(q_abs_a.numel()))

            bias_chunk = torch.zeros((B, q1 - q0, T), device=device, dtype=torch.float32)
            if self.wavelet_logit_bias_debug_assert:
                assert bias_chunk.shape == (B, q1 - q0, T)
            nonnull_gate_chunk = None
            pi_scale_for_sum = pi_scale[:, q0:q1, :]
            if multiscale_rms_after_sum:
                # For post-sum RMS, keep the null/non-null gate outside the RMS.
                # First sum conditional scale contributions, RMS over full context,
                # then multiply by the post-intervention non-null gate.
                nonnull_gate_chunk = nonnull_gate[:, q0:q1, :]
                pi_scale_for_sum = pi_scale_without_null_gate[:, q0:q1, :]
            if use_mlp_bias_baseline:
                # Param-matched non-wavelet baseline: low-rank U@V^T without pi-mixture.
                u_q = torch.tanh(router_logits[:, q0:q1, 1:] / tau)
                beta_ref = beta_m[:, q0:q1]
                if use_scale_coupled_shift:
                    beta_scale = torch.tanh(beta_ref)
                else:
                    beta_scale = torch.tanh(beta_ref / float(max(1, T - 1)))
                u_q = u_q * (1.0 + 0.1 * beta_scale.unsqueeze(-1))
                key_pos = (diff / float(max(1, T - 1))).view(T, 1)
                v_k = self.mlp_bias_basis_mlp(key_pos)
                v_k = self.mlp_bias_basis_ln(v_k)
                v_k = self._rms_norm_last_dim(v_k, eps=eps)
                bias_chunk = torch.einsum("bqk,tk->bqt", u_q, v_k)
                if analysis_enabled and analysis_q_local is not None and analysis_q_abs is not None:
                    k_mlp = min(K, int(u_q.shape[-1]), int(v_k.shape[-1]))
                    for i in range(k_mlp):
                        contrib_i = u_q[..., i].unsqueeze(-1) * v_k[:, i].view(1, 1, T)
                        if far_mask is not None:
                            contrib_i = contrib_i * far_mask.to(dtype=torch.float32)
                        _analysis_accumulate_component(
                            int(i),
                            contrib_i,
                            beta_ref,
                            analysis_q_local,
                            analysis_q_abs,
                        )
            else:
                perm = None
                if basis_control == "permute_scales":
                    perm = self._wavelet_basis_perm(K=K, layer_idx=lid, device=device)
                use_relative_position = bool(getattr(self, "wavelet_ctxscale_use_relative_position", False))
                center_pos_ratio = float(getattr(self, "wavelet_ctxscale_center_pos_ratio", 0.0))
                center_pos_ratio = max(0.0, min(1.0, center_pos_ratio))
                key_anchor_pos = None
                if use_relative_position:
                    # Relative coordinate per query row: delta(q,k) = q_abs - k_abs.
                    # This keeps wavelet basis aligned to query-centric distance instead of absolute key index.
                    q_abs_chunk = torch.arange(q0, q1, device=device, dtype=torch.float32).view(1, q1 - q0, 1)
                elif center_pos_ratio > 0.0:
                    # Key-anchor center follows query position: center(q)=ratio*q.
                    # We then bucket half-offsets by signed ceil so ratio=0.5 gives:
                    # q=2 -> [-1,0,1], q=3 -> [-2,-1,1,2] on causal keys.
                    q_abs_chunk = torch.arange(q0, q1, device=device, dtype=torch.float32).view(1, q1 - q0, 1)
                    centered = diff.view(1, 1, T) - (center_pos_ratio * q_abs_chunk)
                    key_anchor_pos = torch.sign(centered) * torch.ceil(centered.abs())
                for i in range(K):
                    scale_idx = int(perm[i].item()) if perm is not None else int(i)
                    s_i = scales[scale_idx]
                    # PAT-244: beta_m is [B,T,K] when wavelet_ctxscale_shift_per_scale is
                    # enabled (each scale index has its own independently-learned shift
                    # decision); otherwise [B,T] (one shared decision broadcast to all K).
                    beta_m_i = beta_m[:, q0:q1, scale_idx] if shift_per_scale else beta_m[:, q0:q1]
                    if use_scale_coupled_shift:
                        beta_i = beta_m_i * s_i
                    else:
                        beta_i = beta_m_i

                    if bool(getattr(self, "_capture_debug_tensors", True)):
                        # PAT-254 Preflight: real per-query learned shift (beta_i) and
                        # the scale/absolute-key coordinate actually used to build this
                        # scale's basis_table, for reconstructing the true (not
                        # idealized) wavelet template offline. [B,Tq] per scale index.
                        if not hasattr(self, "_last_ctxscale_beta_by_scale") or self._last_ctxscale_beta_by_scale is None:
                            self._last_ctxscale_beta_by_scale = {}
                        self._last_ctxscale_beta_by_scale[int(scale_idx)] = beta_i.detach().to(torch.float32)
                        self._last_ctxscale_diff = diff.detach().to(torch.float32)
                        self._last_ctxscale_scales = scales.detach().to(torch.float32)

                    base_x_i = diff.view(1, 1, T)
                    if getattr(self, "wavelet_ctxscale_dual_center_enable", False):
                        q_abs_chunk = torch.arange(q0, q1, device=device, dtype=torch.float32).view(1, q1 - q0, 1)
                        abs_u_i = (base_x_i - beta_i.unsqueeze(-1)) / s_i
                        query_u_i = ((q_abs_chunk - base_x_i) - beta_i.unsqueeze(-1)) / s_i
                        if self.bias_type != "wavelet":
                            raise ValueError("dual-center wavelet basis requires bias_type='wavelet'")
                        abs_basis = self._ricker_wavelet(abs_u_i)
                        query_basis = self._ricker_wavelet(query_u_i)
                        dual_center_norm_mode = getattr(
                            self, "wavelet_ctxscale_dual_center_norm_mode", "sum_then_rms"
                        )
                        if dual_center_norm_mode in ("separate_rms", "separate_rms_nosqrt"):
                            if getattr(self, "wavelet_logit_bias_center", False):
                                _qc = abs_basis.shape[-2]
                                _Tk = abs_basis.shape[-1]
                                _qabs = torch.arange(q0, q0 + _qc, device=abs_basis.device).view(1, _qc, 1)
                                _kidx = torch.arange(_Tk, device=abs_basis.device).view(1, 1, _Tk)
                                _causal = (_kidx <= _qabs).to(abs_basis.dtype)
                                _cnt = _causal.sum(dim=-1, keepdim=True).clamp_min(1.0)
                                abs_basis = abs_basis - (abs_basis * _causal).sum(dim=-1, keepdim=True) / _cnt
                                query_basis = query_basis - (query_basis * _causal).sum(dim=-1, keepdim=True) / _cnt
                            if not getattr(self, "wavelet_logit_bias_norm_disable", False):
                                abs_basis = self._rms_norm_wavelet_basis(abs_basis, q0=q0, eps=eps)
                                query_basis = self._rms_norm_wavelet_basis(query_basis, q0=q0, eps=eps)
                            if dual_center_norm_mode == "separate_rms_nosqrt":
                                basis_table = abs_basis + query_basis
                            else:
                                basis_table = (abs_basis + query_basis) / math.sqrt(2.0)
                            skip_common_basis_center_norm = True
                        else:
                            basis_table = abs_basis + query_basis
                            skip_common_basis_center_norm = False
                    else:
                        skip_common_basis_center_norm = False
                        if use_relative_position:
                            coord_x_i = q_abs_chunk - base_x_i
                        elif key_anchor_pos is not None:
                            coord_x_i = key_anchor_pos
                        else:
                            coord_x_i = base_x_i
                        token_x_i = coord_x_i - beta_i.unsqueeze(-1)
                        u_i = token_x_i / s_i

                        if self.bias_type == "wavelet":
                            basis_table = self._ricker_wavelet(u_i)
                            basis_table = self._apply_ricker_pl4_pattern(basis_table, base_x_i, beta_i, s_i)
                        elif self.bias_type == "sine":
                            basis_table = self._sine_basis(u_i)
                        elif self.bias_type == "morlet":
                            basis_table = self._morlet_basis(u_i)
                        elif self.bias_type == "ricker_cos":
                            basis_table = self._ricker_cos_basis(u_i)
                        elif self.bias_type == "morlet_gaussamp":
                            basis_table = self._morlet_basis(u_i)
                        elif self.bias_type == "gaussian":
                            basis_table = self._gaussian_basis(u_i)
                        elif self.bias_type == "linear":
                            basis_table = self._linear_basis(u_i)
                        elif self.bias_type == "rotary":
                            pass
                        else:
                            raise ValueError(f"Unsupported bias_type: {self.bias_type}")

                    if (
                        not skip_common_basis_center_norm
                        and not getattr(self, "wavelet_logit_bias_norm_disable", False)
                        and not skip_per_scale_basis_norm
                    ):
                        basis_table = self._rms_norm_wavelet_basis(basis_table, q0=q0, eps=eps)
                    if getattr(self, "_pat234_cap", None) is not None:  # PAT-234 stage probe (default off)
                        self._pat234_cap.setdefault("S1_postnorm", []).append((int(lid), int(scale_idx), int(q0), basis_table.detach().float().cpu()))

                    contrib_i = pi_scale_for_sum[..., i].unsqueeze(-1) * basis_table ### weight * wavelet basis

                    if getattr(self, "_pat234_cap", None) is not None:  # PAT-234: post-gain per scale
                        self._pat234_cap.setdefault("S3_postgain", []).append((int(lid), int(scale_idx), int(q0), contrib_i.detach().float().cpu()))
                    if far_mask is not None:
                        contrib_i = contrib_i * far_mask.to(dtype=torch.float32)
                    bias_chunk = bias_chunk + contrib_i
                    if analysis_enabled and analysis_q_local is not None and analysis_q_abs is not None:
                        _analysis_accumulate_component(
                            int(scale_idx),
                            contrib_i,
                            beta_i,
                            analysis_q_local,
                            analysis_q_abs,
                        )
                if need_log:
                    bias_pre_norm = bias_chunk.detach().to(torch.float32)
                    norm_pre_sum_sq = norm_pre_sum_sq + bias_pre_norm.square().sum()
                    norm_elem_count += int(bias_pre_norm.numel())
                if self.wavelet_ctxscale_amplitude_multiplier_override:
                    bias_chunk = bias_chunk * self.wavelet_ctxscale_amplitude_multiplier
                    if need_log:
                        norm_scale_vals.append(
                            torch.tensor(
                                [self.wavelet_ctxscale_amplitude_multiplier],
                                device=device,
                                dtype=torch.float32,
                            )
                        )
                elif multiscale_rms_after_sum:
                    # PAT-225 causal/window-cap ablation: 0 keeps prior full-width RMS behavior.
                    bias_chunk_f = bias_chunk.to(torch.float32)
                    multiscale_denom = torch.sqrt(
                        self._windowed_mean_sq(
                            bias_chunk_f,
                            int(getattr(self, "wavelet_ctxscale_rms_train_window", 0)),
                        ).clamp_min(0.0)
                        + float(eps)
                    )
                    bias_chunk = (bias_chunk_f / multiscale_denom).to(
                        dtype=bias_chunk.dtype
                    )
                    if nonnull_gate_chunk is not None:
                        bias_chunk = bias_chunk * nonnull_gate_chunk.to(
                            dtype=bias_chunk.dtype
                        )
                    if need_log:
                        norm_scale_vals.append(
                            multiscale_denom.detach()
                            .reciprocal()
                            .to(torch.float32)
                            .reshape(-1)
                        )
                elif self.multiscale_norm_requested in (
                    "sqrt_keff_detach",
                    "keff_detach",
                ):
                    self._validate_dynamic_multiscale_norm_router(
                        self.multiscale_norm_requested,
                        router_mode,
                        intervention_active=do_active,
                    )
                    # Use independent scale gates only; the null gate is excluded.
                    g_chunk = g[:, q0:q1, :]
                    multiscale_scale = self._get_sqrt_keff_detach_scale(
                        g_chunk,
                        eps=eps,
                        K=self.wavelet_ctxscale_k,
                    )
                    if need_log:
                        scale_detached = multiscale_scale.detach().to(torch.float32)
                        norm_scale_vals.append(scale_detached.reshape(-1))
                        norm_keff_vals.append(
                            scale_detached.square().reciprocal().reshape(-1)
                        )
                    bias_chunk = bias_chunk * multiscale_scale.to(
                        dtype=bias_chunk.dtype
                    )
                else:
                    bias_chunk = bias_chunk * self.multiscale_sum_scale
                    if need_log:
                        norm_scale_vals.append(
                            torch.tensor(
                                [self.multiscale_sum_scale],
                                device=device,
                                dtype=torch.float32,
                            )
                        )
                if need_log:
                    norm_post_sum_sq = (
                        norm_post_sum_sq
                        + bias_chunk.detach().to(torch.float32).square().sum()
                    )
            if far_mask is not None:
                bias_chunk = bias_chunk * far_mask.to(dtype=torch.float32)
            if far_over_mask is not None:
                over_m = far_over_mask.to(dtype=torch.float32)
                keep_m = 1.0 - over_m
                bias_chunk = bias_chunk * keep_m + bias_chunk * over_m * float(far_over_alpha)

            if enable_film:
                pi_chunk = pi[:, q0:q1, :]
                rho_chunk = rho[:, q0:q1]
                bias_chunk, s_raw, t_raw, scale_m, shift_m = self._ctxscale_apply_bias_film(
                    bias_chunk=bias_chunk,
                    pi_chunk=pi_chunk,
                    rho_chunk=rho_chunk,
                    g_layer=g_layer,
                    use_mlp=use_mlp_bias_baseline,
                )
                film_nf_flags["s_raw"] |= int((~torch.isfinite(s_raw)).any().item())
                film_nf_flags["t_raw"] |= int((~torch.isfinite(t_raw)).any().item())
                film_nf_flags["scale"] |= int((~torch.isfinite(scale_m)).any().item())
                film_nf_flags["shift"] |= int((~torch.isfinite(shift_m)).any().item())
                if self.wavelet_logit_bias_debug_assert:
                    if not torch.isfinite(scale_m).all():
                        raise FloatingPointError("ctxscale_shift_v0: non-finite film scale")
                    if not torch.isfinite(shift_m).all():
                        raise FloatingPointError("ctxscale_shift_v0: non-finite film shift")
                sat_s_num += int((s_raw.detach().abs() > 7.5).sum().item())
                sat_t_num += int((t_raw.detach().abs() > 7.5).sum().item())
                sat_den += int(s_raw.numel())
            g_bias_max = float(getattr(self.config, "wavelet_ctxscale_g_bias_max", self.wavelet_ctxscale_g_bias_max))
            if use_head_gate and g_head is not None:
                eff_to_add = bias_chunk.unsqueeze(1) * g_head.view(1, -1, 1, 1)
                eff_to_add = eff_to_add.clamp(min=-g_bias_max, max=g_bias_max)
                if not torch.isfinite(eff_to_add).all():
                    _warn_nonfinite("g_bias_head")
                    eff_to_add = torch.nan_to_num(
                        eff_to_add,
                        nan=0.0,
                        posinf=g_bias_max,
                        neginf=-g_bias_max,
                    )
                eff_chunk = eff_to_add.mean(dim=1)
            else:
                eff_chunk = g_layer * bias_chunk
                if getattr(self, "_pat234_cap", None) is not None:  # PAT-234: pre final g_bias clamp
                    self._pat234_cap.setdefault("S4pre_preclamp", []).append((int(lid), int(q0), eff_chunk.detach().float().cpu()))
                eff_chunk = eff_chunk.clamp(min=-g_bias_max, max=g_bias_max)
                if getattr(self, "_pat234_cap", None) is not None:  # PAT-234: post final g_bias clamp
                    self._pat234_cap.setdefault("S4post_postclamp", []).append((int(lid), int(q0), eff_chunk.detach().float().cpu()))
                if not torch.isfinite(eff_chunk).all():
                    _warn_nonfinite("g_bias")
                    eff_chunk = torch.nan_to_num(
                        eff_chunk,
                        nan=0.0,
                        posinf=g_bias_max,
                        neginf=-g_bias_max,
                    )
                eff_to_add = eff_chunk.unsqueeze(1)
            if head_mask is not None:
                eff_to_add = eff_to_add * head_mask

            if do_active and (0 <= int(do_target_head) < int(eff_to_add.shape[1])):
                do_t = eff_to_add[:, int(do_target_head), :, :].detach().float()
                if int(do_t.numel()) > 0:
                    do_target_sum_sq += float(do_t.square().sum().item())
                    do_target_sum += float(do_t.sum().item())
                    do_target_count += int(do_t.numel())
                    do_target_maxabs = max(do_target_maxabs, float(do_t.abs().max().item()))
                for nh, acc in do_non_target_acc.items():
                    if 0 <= int(nh) < int(eff_to_add.shape[1]):
                        do_n = eff_to_add[:, int(nh), :, :].detach().float()
                        if int(do_n.numel()) > 0:
                            acc["sum_sq"] += float(do_n.square().sum().item())
                            acc["sum"] += float(do_n.sum().item())
                            acc["count"] += int(do_n.numel())
                            acc["maxabs"] = max(float(acc["maxabs"]), float(do_n.abs().max().item()))

            # A0 lambda sweep: scale QWAB bias at inference (wavelet_ctxscale_lambda != 1.0)
            _lambda = float(getattr(self.config, "wavelet_ctxscale_lambda", 1.0))
            if _lambda != 1.0:
                # row-center then scale: softmax is invariant to row-wise constant shifts
                _row_mean = eff_to_add.mean(dim=-1, keepdim=True)
                eff_to_add = (eff_to_add - _row_mean) * _lambda
            _qwab_dampen_threshold = int(getattr(self, "wavelet_qwab_beyond_dampen_threshold", 0))
            _qwab_dampen_alpha = float(getattr(self, "wavelet_qwab_beyond_dampen_alpha", 1.0))
            if _qwab_dampen_threshold > 0 and _qwab_dampen_alpha != 1.0:
                _q_pos_chunk = torch.arange(q0, q1, device=eff_to_add.device, dtype=torch.long).view(1, 1, -1, 1)
                _k_pos_chunk = torch.arange(eff_to_add.shape[-1], device=eff_to_add.device, dtype=torch.long).view(1, 1, 1, -1)
                _qwab_beyond_mask = (_q_pos_chunk - _k_pos_chunk) >= _qwab_dampen_threshold
                eff_to_add = torch.where(_qwab_beyond_mask, eff_to_add * _qwab_dampen_alpha, eff_to_add)
            _qwab_within_threshold = int(getattr(self, "wavelet_qwab_within_dampen_threshold", 0))
            _qwab_within_alpha = float(getattr(self, "wavelet_qwab_within_dampen_alpha", 1.0))
            if _qwab_within_threshold > 0 and _qwab_within_alpha != 1.0:
                _q_pos_chunk2 = torch.arange(q0, q1, device=eff_to_add.device, dtype=torch.long).view(1, 1, -1, 1)
                _k_pos_chunk2 = torch.arange(eff_to_add.shape[-1], device=eff_to_add.device, dtype=torch.long).view(1, 1, 1, -1)
                _qwab_within_mask = (_q_pos_chunk2 - _k_pos_chunk2) < _qwab_within_threshold
                eff_to_add = torch.where(_qwab_within_mask, eff_to_add * _qwab_within_alpha, eff_to_add)
            bias_eval_mult = float(getattr(self, "wavelet_logit_bias_eval_mult", 1.0))
            if bool(getattr(self, "wavelet_ctxscale_gain_st", False)) and bias_eval_mult != 1.0:
                scaled = eff_to_add * bias_eval_mult
                eff_to_add_final = eff_to_add + (scaled - eff_to_add).detach()
            else:
                eff_to_add_final = eff_to_add * bias_eval_mult
            logits_out[:, :, q0:q1, :] = logits_out[:, :, q0:q1, :] + eff_to_add_final

            if analysis_enabled and analysis_q_local is not None and analysis_q_abs is not None and analysis_eff_abs_vals is not None:
                for j in range(int(analysis_q_local.numel())):
                    q_rel = int(analysis_q_local[j].item())
                    q_abs_val = int(analysis_q_abs[j].item())
                    k_idx = _analysis_k_idx_for_q(q_abs_val)
                    if k_idx is None or int(k_idx.numel()) == 0:
                        continue
                    e_vals = eff_chunk[:, q_rel, :].index_select(-1, k_idx).abs().reshape(-1)
                    if int(e_vals.numel()) > 0:
                        analysis_eff_abs_vals.append(e_vals.detach().cpu())

            if need_log and sample_q_idx is not None and sample_count < sample_budget:
                local = (sample_q_idx >= q0) & (sample_q_idx < q1)
                if bool(local.any()):
                    q_abs = sample_q_idx[local]
                    q_local = q_abs - q0
                    b_sel = bias_chunk.index_select(1, q_local).index_select(2, sample_k_idx)
                    e_sel = eff_chunk.index_select(1, q_local).index_select(2, sample_k_idx)
                    base_chunk = E_base_raw[:, :, q0:q1, :].mean(dim=1)
                    base_sel = base_chunk.index_select(1, q_local).index_select(2, sample_k_idx)
                    valid = sample_k_idx.view(1, 1, -1) <= q_abs.view(1, -1, 1)
                    valid = valid.expand(B, -1, -1)
                    b_vals = b_sel[valid]
                    e_vals = e_sel[valid]
                    base_vals = base_sel[valid]
                    if b_vals.numel() > 0:
                        take = min(sample_budget - sample_count, int(b_vals.numel()))
                        sample_bias_vals.append(b_vals[:take].detach())
                        sample_eff_vals.append(e_vals[:take].detach())
                        sample_base_vals.append(base_vals[:take].detach())
                        if enable_film:
                            s_sel = scale_m.index_select(1, q_local).reshape(-1)
                            t_sel = shift_m.index_select(1, q_local).reshape(-1)
                            sr_sel = s_raw.index_select(1, q_local).reshape(-1)
                            tr_sel = t_raw.index_select(1, q_local).reshape(-1)
                            sample_scale_vals.append(s_sel[:take].detach())
                            sample_shift_vals.append(t_sel[:take].detach())
                            sample_sraw_vals.append(sr_sel[:take].detach())
                            sample_traw_vals.append(tr_sel[:take].detach())
                        sample_count += take

            if self.wavelet_logit_bias_debug_assert:
                if not torch.isfinite(bias_chunk).all():
                    raise FloatingPointError("ctxscale_shift_v0: non-finite bias chunk")
                if not torch.isfinite(eff_chunk).all():
                    raise FloatingPointError("ctxscale_shift_v0: non-finite effective bias chunk")

        self._last_logits_full = logits_out.detach().to(dtype=torch.float32)
        if self.wavelet_logit_bias_debug_assert and not torch.isfinite(logits_out).all():
            raise FloatingPointError("ctxscale_shift_v0: non-finite logits after bias injection")

        if do_active:
            target_rms = 0.0
            target_mean = 0.0
            if do_target_count > 0:
                target_rms = float((do_target_sum_sq / float(do_target_count)) ** 0.5)
                target_mean = float(do_target_sum / float(do_target_count))
            non_target_delta = {}
            for nh, acc in do_non_target_acc.items():
                c = int(acc.get("count", 0))
                if c > 0:
                    rms_n = float((float(acc["sum_sq"]) / float(c)) ** 0.5)
                    mean_n = float(float(acc["sum"]) / float(c))
                else:
                    rms_n = 0.0
                    mean_n = 0.0
                non_target_delta[str(int(nh))] = {
                    "rms": float(rms_n),
                    "maxabs": float(acc.get("maxabs", 0.0)),
                    "mean": float(mean_n),
                    "count": int(c),
                }

            do_validation = {
                "enabled": True,
                "layer": int(lid),
                "mode": str(do_mode),
                "scope": str(do_scope),
                "num_scales": int(K),
                "target_heads": [int(x) for x in do_target_heads],
                "target_head_for_check": int(do_target_head),
                "target_pi": {
                    "min": float(do_stat.get("target_pi_min", 0.0)),
                    "max": float(do_stat.get("target_pi_max", 0.0)),
                    "pi0_mean": float(do_stat.get("target_pi0_mean", 0.0)),
                    "sum_mean": float(do_stat.get("target_sum_prob_mean", 0.0)),
                    "sum_min": float(do_stat.get("target_sum_prob_min", 0.0)),
                    "sum_max": float(do_stat.get("target_sum_prob_max", 0.0)),
                    "argmax_expected_frac": float(do_stat.get("target_argmax_expected_frac", 0.0)),
                    "nonzero_count_mean": float(do_stat.get("target_nonzero_count_mean", 0.0)),
                    "nonzero_count_max": float(do_stat.get("target_nonzero_count_max", 0.0)),
                    "uniform_scale_maxdev": float(do_stat.get("target_uniform_scale_maxdev", 0.0)),
                    "expected_argmax": int(do_stat.get("expected_argmax", -1)),
                    "scale_idx": int(do_stat.get("scale_idx", -1)),
                },
                "target_delta_wavelet": {
                    "rms": float(target_rms),
                    "maxabs": float(do_target_maxabs),
                    "mean": float(target_mean),
                    "count": int(do_target_count),
                },
                "non_target_pi_change": {
                    "maxabs": float(do_stat.get("non_target_pi_change_maxabs", 0.0)),
                    "meanabs": float(do_stat.get("non_target_pi_change_meanabs", 0.0)),
                },
                "non_target_delta_wavelet": non_target_delta,
            }
            strict_check = bool(int(do_stat.get("strict", 0)))
            if strict_check and str(do_mode) == "null":
                # For do(null), wavelet branch must be effectively disabled for target head.
                if float(target_rms) > 1e-8 or float(do_target_maxabs) > 1e-8:
                    raise AssertionError(
                        f"do(null) ΔE validation failed at layer={int(lid)} head={int(do_target_head)}: "
                        f"rms={float(target_rms):.6e}, maxabs={float(do_target_maxabs):.6e}"
                    )
            self._last_ctxscale_do_validation = do_validation

        if analysis_enabled and analysis_sample_q_idx is not None:
            pi_sample_a = torch.nan_to_num(
                pi.index_select(1, analysis_sample_q_idx).detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            pi_dist_a = _router_diag_prob_dist(pi_sample_a)
            pi_entropy_a = -(pi_dist_a * pi_dist_a.log()).sum(dim=-1)
            pi_top1_a = pi_dist_a.max(dim=-1).values
            pi_top2_a = torch.topk(pi_dist_a, k=2, dim=-1).values
            pi_margin_a = (pi_top2_a[..., 0] - pi_top2_a[..., 1]).clamp_min(0.0)
            pi_null_a = pi_dist_a[..., 0]
            pi_entropy_q_a = _quantiles_flat(pi_entropy_a, qs=(0.5, 0.9))
            pi_top1_q_a = _quantiles_flat(pi_top1_a, qs=(0.9,))
            pi_margin_q_a = _quantiles_flat(pi_margin_a, qs=(0.5, 0.9))
            pi_mean = pi_dist_a.mean(dim=tuple(range(pi_dist_a.ndim - 1)))

            beta_sample_a = torch.nan_to_num(
                beta_m.index_select(1, analysis_sample_q_idx).detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if shift_per_scale:
                # PAT-244: same trailing-K-dim issue as beta_sample above -- this
                # analysis block predates per-scale shift and expects [B,T].
                beta_sample_a = beta_sample_a.mean(dim=-1)
            beta_over_t_a = beta_sample_a / float(max(1, int(T - 1)))
            beta_over_t_q_a = _quantiles_flat(beta_over_t_a, qs=(0.5, 0.9, 0.99))
            if use_scale_coupled_shift:
                shift_unit_max_a = max(float(getattr(self, "wavelet_ctxscale_shift_unit_max", 1.0)), 1e-6)
                beta_clamped_a = float((beta_sample_a.abs() >= 0.99 * shift_unit_max_a).float().mean().item())
            else:
                if use_abs_shift_causal:
                    q_pos_sample_a = analysis_sample_q_idx.to(device=device, dtype=torch.float32).view(1, -1)
                    beta_clamped_a = float((beta_sample_a >= q_pos_sample_a).float().mean().item())
                else:
                    beta_upper_a = float(max(1, int(T_used - 1)) if apply_shift_t_scaling else max(1, int(T_test - 1)))
                    beta_clamped_a = float((beta_sample_a >= beta_upper_a).float().mean().item())
            if K > 0:
                pi_scale_a = pi_dist_a[..., 1 : (K + 1)]
                scale_shape = [1] * (pi_scale_a.dim() - 1) + [K]
                scale_vec = scales.view(*scale_shape)
                scale_denom = pi_scale_a.sum(dim=-1).clamp_min(1e-12)
                exp_scale = (pi_scale_a * scale_vec).sum(dim=-1) / scale_denom
                top1_idx = pi_scale_a.argmax(dim=-1)
                top1_scale = scales.index_select(0, top1_idx.reshape(-1)).view_as(top1_idx)
                beta_sample_expand = beta_sample_a
                while beta_sample_expand.dim() < top1_scale.dim():
                    beta_sample_expand = beta_sample_expand.unsqueeze(-1)
                if use_scale_coupled_shift:
                    beta_token_top1 = beta_sample_expand * top1_scale
                    beta_token_exp = beta_sample_expand * exp_scale
                else:
                    beta_token_top1 = beta_sample_expand
                    beta_token_exp = beta_sample_expand
                beta_over_s_top1 = beta_token_top1 / top1_scale.clamp_min(1e-6)
                beta_over_s_exp = beta_token_exp / exp_scale.clamp_min(1e-6)
                beta_over_s_top1_q = _quantiles_flat(beta_over_s_top1, qs=(0.5, 0.9, 0.99))
                beta_over_s_exp_q = _quantiles_flat(beta_over_s_exp, qs=(0.5, 0.9, 0.99))
            else:
                beta_over_s_top1_q = {"p50": 0.0, "p90": 0.0, "p99": 0.0}
                beta_over_s_exp_q = {"p50": 0.0, "p90": 0.0, "p99": 0.0}

            eff_bias_abs_p99 = 0.0
            if analysis_eff_abs_vals is not None and len(analysis_eff_abs_vals) > 0:
                eff_abs_cat = torch.cat(analysis_eff_abs_vals, dim=0).float()
                eff_q = _quantiles_flat(eff_abs_cat, qs=(0.99,))
                eff_bias_abs_p99 = float(eff_q["p99"])

            energy_i = []
            absmean_i = []
            width_mean_i = []
            width_p50_i = []
            width_p90_i = []
            for i in range(K):
                c_i = float(analysis_count[i]) if analysis_count is not None else 0.0
                if c_i > 0.0:
                    energy_i.append(float(analysis_energy_sum[i] / c_i))
                    absmean_i.append(float(analysis_abs_sum[i] / c_i))
                else:
                    energy_i.append(0.0)
                    absmean_i.append(0.0)
                w_vals = analysis_width_vals[i] if analysis_width_vals is not None else []
                if len(w_vals) == 0:
                    width_mean_i.append(0.0)
                    width_p50_i.append(0.0)
                    width_p90_i.append(0.0)
                else:
                    w_t = torch.tensor(w_vals, dtype=torch.float32)
                    w_q = _quantiles_flat(w_t, qs=(0.5, 0.9))
                    width_mean_i.append(float(w_t.mean().item()))
                    width_p50_i.append(float(w_q["p50"]))
                    width_p90_i.append(float(w_q["p90"]))

            self._wavelet_analysis_write_record(
                layer_idx=lid,
                step=step,
                payload={
                    "wavelet_mode": str(wavelet_mode_resolved),
                    "basis_control": str(basis_control),
                    "router_mode": str(router_mode),
                    "router_jitter_style": str(router_jitter_style),
                    "router_jitter_style_resolved": str(router_jitter_style_resolved),
                    "router_jitter_std": float(router_jitter_std),
                    "router_jitter_enabled": int(router_jitter_enabled),
                    "router_jitter_injected": int(router_jitter_injected),
                    "router_jitter_target_flip_probability": float(router_jitter_target_flip_probability),
                    "use_relative_position": int(bool(getattr(self, "wavelet_ctxscale_use_relative_position", False))),
                    "center_pos_ratio": float(getattr(self, "wavelet_ctxscale_center_pos_ratio", 0.0)),
                    "dual_center": int(bool(getattr(self, "wavelet_ctxscale_dual_center_enable", False))),
                    "pattern_mode": str(getattr(self, "wavelet_ctxscale_pattern_mode", "ricker")),
                    "restore_bin": int(getattr(self, "wavelet_ctxscale_restore_bin", -1)),
                    "scale_values": [float(scales[i].item()) for i in range(K)],
                    "scales": [float(scales[i].item()) for i in range(K)],
                    "K": int(K),
                    "pi_mean": [float(x.item()) for x in pi_mean],
                    "energy_i": energy_i,
                    "absmean_i": absmean_i,
                    "bias_energy_per_scale": energy_i,
                    "bias_absmean_per_scale": absmean_i,
                    "width_mean_i": width_mean_i,
                    "width_p50_i": width_p50_i,
                    "width_p90_i": width_p90_i,
                    "width_mean_per_scale": width_mean_i,
                    "width_p50_per_scale": width_p50_i,
                    "width_p90_per_scale": width_p90_i,
                    "pi_entropy_mean": float(pi_entropy_a.mean().item()),
                    "pi_entropy_p50": float(pi_entropy_q_a["p50"]),
                    "pi_entropy_p90": float(pi_entropy_q_a["p90"]),
                    "pi_top1_mean": float(pi_top1_a.mean().item()),
                    "pi_top1_p90": float(pi_top1_q_a["p90"]),
                    "pi_margin_mean": float(pi_margin_a.mean().item()),
                    "pi_margin_p50": float(pi_margin_q_a["p50"]),
                    "pi_margin_p90": float(pi_margin_q_a["p90"]),
                    "pi_null_mean": float(pi_null_a.mean().item()),
                    "null_mean": float(pi_null_a.mean().item()),
                    "sigma_mean": float(router_jitter_sigma_mean),
                    "sigma_std": float(router_jitter_sigma_std),
                    "flip_probability_estimate": float(router_jitter_flip_probability_estimate),
                    "router_entropy_reg_loss": float(router_entropy_reg_loss.detach().item()),
                    "router_entropy_reg_active_frac": float(router_entropy_reg_active_frac.item()),
                    "beta_over_T_p50": float(beta_over_t_q_a["p50"]),
                    "beta_over_T_p90": float(beta_over_t_q_a["p90"]),
                    "beta_over_T_p99": float(beta_over_t_q_a["p99"]),
                    "beta_over_s_top1_p50": float(beta_over_s_top1_q["p50"]),
                    "beta_over_s_top1_p90": float(beta_over_s_top1_q["p90"]),
                    "beta_over_s_top1_p99": float(beta_over_s_top1_q["p99"]),
                    "beta_over_s_exp_p50": float(beta_over_s_exp_q["p50"]),
                    "beta_over_s_exp_p90": float(beta_over_s_exp_q["p90"]),
                    "beta_over_s_exp_p99": float(beta_over_s_exp_q["p99"]),
                    "g_layer": float(g_layer.detach().float().mean().item()),
                    "g_layer_raw": float(g_layer_raw.detach().float().mean().item()) if torch.is_tensor(g_layer_raw) else float(g_layer_raw),
                    "grad_abs": float(gate_state.get("grad_abs", float("nan"))),
                    "sat_low": int(gate_state.get("sat_low", 0)),
                    "sat_high": int(gate_state.get("sat_high", 0)),
                    "sat_extreme": int(gate_state.get("sat_extreme", 0)),
                    "clamp_frac": float(beta_clamped_a),
                    "eff_bias_abs_p99": float(eff_bias_abs_p99),
                    "q_sample_count": int(analysis_q_count),
                    "qk_sample_count": int(analysis_qk_count),
                    "batch_size": int(B),
                    "seq_len": int(T),
                    "bucket_T": int(T),
                },
            )

        payload = None
        if need_log:
            sum_g_mean = float("nan")
            sum_g_p90 = float("nan")
            g0_mean = float("nan")
            g0_p90 = float("nan")
            alpha_mean = float("nan")
            alpha_p90 = float("nan")
            pi_sample = torch.nan_to_num(pi.index_select(1, sample_q_idx).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            pi_dist = _router_diag_prob_dist(pi_sample)
            pi_entropy = -(pi_dist * pi_dist.log()).sum(dim=-1)
            pi_top1 = pi_dist.max(dim=-1).values
            pi_top2 = torch.topk(pi_dist, k=2, dim=-1).values
            pi_margin = (pi_top2[..., 0] - pi_top2[..., 1]).clamp_min(0.0)
            pi_null = pi_dist[..., 0]
            pi_entropy_q = _quantiles_flat(pi_entropy, qs=(0.5, 0.9))
            pi_top1_q = _quantiles_flat(pi_top1, qs=(0.9,))
            pi_margin_q = _quantiles_flat(pi_margin, qs=(0.5, 0.9))
            if sum_g is not None:
                sum_g_sample = torch.nan_to_num(
                    sum_g.squeeze(-1).index_select(1, sample_q_idx).detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                sum_g_q = _quantiles_flat(sum_g_sample, qs=(0.9,))
                sum_g_mean = float(sum_g_sample.mean().item())
                sum_g_p90 = float(sum_g_q["p90"])
            if g0_gate is not None:
                g0_sample = torch.nan_to_num(
                    g0_gate.squeeze(-1).index_select(1, sample_q_idx).detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                g0_q = _quantiles_flat(g0_sample, qs=(0.9,))
                g0_mean = float(g0_sample.mean().item())
                g0_p90 = float(g0_q["p90"])
            if alpha_gate is not None:
                alpha_sample = torch.nan_to_num(
                    alpha_gate.squeeze(-1).index_select(1, sample_q_idx).detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                alpha_q = _quantiles_flat(alpha_sample, qs=(0.9,))
                alpha_mean = float(alpha_sample.mean().item())
                alpha_p90 = float(alpha_q["p90"])

            rho_sample = torch.nan_to_num(rho.index_select(1, sample_q_idx).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if shift_per_scale:
                # PAT-244: rho/beta_m carry a trailing K dim when per-scale shift is
                # enabled; this diagnostic block predates that and expects [B,T].
                # Mean over K for logging purposes only -- does not affect training.
                rho_sample = rho_sample.mean(dim=-1)
            rho_q = _quantiles_flat(rho_sample, qs=(0.5, 0.9, 0.99))

            beta_sample = torch.nan_to_num(beta_m.index_select(1, sample_q_idx).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            beta_per_scale_mean_str = ""
            if shift_per_scale:
                # PAT-244: capture the per-scale breakdown BEFORE collapsing to [B,T]
                # for the (K-agnostic) quantile block below -- otherwise there is no
                # way to see whether the K independently-learned shift heads have
                # actually diverged from each other.
                _beta_per_scale_mean = beta_sample.mean(dim=(0, 1)).tolist()
                beta_per_scale_mean_str = "beta_per_scale_mean=[" + ",".join(f"{v:.4e}" for v in _beta_per_scale_mean) + "] "
                beta_sample = beta_sample.mean(dim=-1)
            beta_q = _quantiles_flat(beta_sample, qs=(0.5, 0.9, 0.99))
            if use_scale_coupled_shift:
                shift_unit_max = max(float(getattr(self, "wavelet_ctxscale_shift_unit_max", 1.0)), 1e-6)
                beta_clamped = float((beta_sample.abs() >= 0.99 * shift_unit_max).float().mean().item())
            else:
                if use_abs_shift_causal:
                    q_pos_sample = sample_q_idx.to(device=device, dtype=torch.float32).view(1, -1)
                    beta_clamped = float((beta_sample >= q_pos_sample).float().mean().item())
                else:
                    beta_upper = float(max(1, int(T_used - 1)) if apply_shift_t_scaling else max(1, int(T_test - 1)))
                    beta_clamped = float((beta_sample >= beta_upper).float().mean().item())
            q_pos_sample = sample_q_idx.to(device=device, dtype=torch.float32).view(1, -1)
            beta_over_q = beta_sample / q_pos_sample.clamp_min(1.0)
            beta_over_q_q = _quantiles_flat(beta_over_q, qs=(0.5, 0.9, 0.99))
            t_used_den = float(max(1, int(T_used - 1)))
            t_test_den = float(max(1, int(T_test - 1)))
            beta_over_tused = beta_sample / t_used_den
            beta_over_ttest = beta_sample / t_test_den
            beta_over_tused_q = _quantiles_flat(beta_over_tused, qs=(0.5, 0.9, 0.99))
            beta_over_ttest_q = _quantiles_flat(beta_over_ttest, qs=(0.5, 0.9, 0.99))

            bias_sample = (
                torch.cat(sample_bias_vals, dim=0)
                if len(sample_bias_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            eff_sample = (
                torch.cat(sample_eff_vals, dim=0)
                if len(sample_eff_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            base_sample = (
                torch.cat(sample_base_vals, dim=0)
                if len(sample_base_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            scale_sample = (
                torch.cat(sample_scale_vals, dim=0)
                if len(sample_scale_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            shift_sample = (
                torch.cat(sample_shift_vals, dim=0)
                if len(sample_shift_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            sraw_sample = (
                torch.cat(sample_sraw_vals, dim=0)
                if len(sample_sraw_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            traw_sample = (
                torch.cat(sample_traw_vals, dim=0)
                if len(sample_traw_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            sat_s = float(sat_s_num / max(1, sat_den))
            sat_t = float(sat_t_num / max(1, sat_den))
            norm_scale_sample = (
                torch.cat(norm_scale_vals, dim=0)
                if len(norm_scale_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            norm_keff_sample = (
                torch.cat(norm_keff_vals, dim=0)
                if len(norm_keff_vals) > 0
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            norm_scale_q = _quantiles_flat(
                norm_scale_sample,
                qs=(0.5, 0.9),
            )
            norm_keff_q = _quantiles_flat(
                norm_keff_sample,
                qs=(0.5, 0.9),
            )
            if norm_elem_count > 0:
                norm_pre_rms = float(
                    torch.sqrt(norm_pre_sum_sq / float(norm_elem_count)).item()
                )
                norm_post_rms = float(
                    torch.sqrt(norm_post_sum_sq / float(norm_elem_count)).item()
                )
            else:
                norm_pre_rms = float("nan")
                norm_post_rms = float("nan")

            payload = {
                "g_layer_raw": g_layer_raw.detach(),
                "g_layer_raw_used": g_layer_raw_used.detach(),
                "g_layer": g_layer.detach(),
                "sig": float(gate_state["sig"]),
                "sat_low": int(gate_state["sat_low"]),
                "sat_high": int(gate_state["sat_high"]),
                "sat_extreme": int(gate_state["sat_extreme"]),
                "grad_abs": float(gate_state["grad_abs"]),
                "grad_abs_p50": float(gate_state.get("grad_abs_p50", float("nan"))),
                "grad_abs_p90": float(gate_state.get("grad_abs_p90", float("nan"))),
                "grad_abs_max": float(gate_state.get("grad_abs_max", float("nan"))),
                "grad_zero_ratio": float(gate_state.get("grad_zero_ratio", float("nan"))),
                "grad_finite_ratio": float(gate_state.get("grad_finite_ratio", float("nan"))),
                "grad_nonfinite": int(gate_state.get("grad_nonfinite", -1)),
                "grad_missing": int(gate_state.get("grad_missing", 1)),
                "grad_zero": int(gate_state["grad_zero"]),
                "delta_a": float(gate_state["delta_a"]),
                "update_ratio": float(gate_state["update_ratio"]),
                "locked": int(gate_state["locked"]),
                "raw_clamped": int(g_layer_raw_clamped),
                "raw_clamp_abs": float(clamp_abs),
                "autofix_active": int(autofix_active),
                "param_req_grad": int(
                    bool(
                        gate_head_param.requires_grad
                        if (use_head_gate and gate_head_param is not None)
                        else gate_param.requires_grad
                    )
                ),
                "pi_entropy_mean": float(pi_entropy.mean().item()),
                "pi_entropy_p50": float(pi_entropy_q["p50"]),
                "pi_entropy_p90": float(pi_entropy_q["p90"]),
                "pi_top1_mean": float(pi_top1.mean().item()),
                "pi_top1_p90": float(pi_top1_q["p90"]),
                "pi_margin_mean": float(pi_margin.mean().item()),
                "pi_margin_p50": float(pi_margin_q["p50"]),
                "pi_margin_p90": float(pi_margin_q["p90"]),
                "pi_null_mean": float(pi_null.mean().item()),
                "rho_p50": float(rho_q["p50"]),
                "rho_p90": float(rho_q["p90"]),
                "rho_p99": float(rho_q["p99"]),
                "beta_p50": float(beta_q["p50"]),
                "beta_p90": float(beta_q["p90"]),
                "beta_p99": float(beta_q["p99"]),
                "beta_clamp_frac": beta_clamped,
                "beta_per_scale_mean_str": beta_per_scale_mean_str,
                "far_only": int(far_only),
                "far_min_delta": int(self.wavelet_ctxscale_far_min_delta),
                "head_frac": float(head_mask.mean().item()) if head_mask is not None else 1.0,
                "head_count": int(len(valid_heads)) if valid_heads is not None else int(E_base_raw.shape[1]),
                "head_gate": int(bool(use_head_gate)),
                "disable_layer_gate": int(bool(disable_layer_gate)),
                "gate_branch": str(gate_branch),
                "cfg_disable_layer_gate": int(bool(getattr(self, "wavelet_ctxscale_disable_layer_gate", False))),
                "cfg_use_head_gate": int(bool(getattr(self, "wavelet_ctxscale_use_head_gate", False))),
                "use_relative_position": int(bool(getattr(self, "wavelet_ctxscale_use_relative_position", False))),
                "center_pos_ratio": float(getattr(self, "wavelet_ctxscale_center_pos_ratio", 0.0)),
                "dual_center": int(bool(getattr(self, "wavelet_ctxscale_dual_center_enable", False))),
                "pattern_mode": str(getattr(self, "wavelet_ctxscale_pattern_mode", "ricker")),
                "restore_bin": int(getattr(self, "wavelet_ctxscale_restore_bin", -1)),
                "scale_coupled_shift": int(use_scale_coupled_shift),
                "abs_shift_causal": int(use_abs_shift_causal),
                "shift_T_mode": str(shift_t_mode),
                "shift_T_ref": int(getattr(self, "wavelet_shift_T_ref", T_used)),
                "shift_T_used": int(T_used),
                "shift_T_test": int(T_test),
                "beta_over_Tused_p50": float(beta_over_tused_q["p50"]),
                "beta_over_Tused_p90": float(beta_over_tused_q["p90"]),
                "beta_over_Tused_p99": float(beta_over_tused_q["p99"]),
                "beta_over_Ttest_p50": float(beta_over_ttest_q["p50"]),
                "beta_over_Ttest_p90": float(beta_over_ttest_q["p90"]),
                "beta_over_Ttest_p99": float(beta_over_ttest_q["p99"]),
                "beta_over_q_p50": float(beta_over_q_q["p50"]),
                "beta_over_q_p90": float(beta_over_q_q["p90"]),
                "beta_over_q_p99": float(beta_over_q_q["p99"]),
                "g_head_mean": float(g_head.mean().item()) if (use_head_gate and g_head is not None) else float("nan"),
                "g_head_p50": float(_quantiles_flat(g_head.detach().float(), qs=(0.5,))["p50"])
                if (use_head_gate and g_head is not None)
                else float("nan"),
                "g_head_p90": float(_quantiles_flat(g_head.detach().float(), qs=(0.9,))["p90"])
                if (use_head_gate and g_head is not None)
                else float("nan"),
                "feat_mode": str(getattr(self, "wavelet_ctx_feat_mode", "q_meanH")),
                "wavelet_mode": str(wavelet_mode_resolved),
                "basis_control": str(basis_control),
                "router_mode": str(router_mode),
                "router_norm_mode": str(getattr(self, "_last_router_norm_mode", "none")),
                "router_cosine": int(getattr(self, "_last_router_cosine", 0)),
                "router_tau_null": float(getattr(self, "_last_router_tau_null", float("nan"))),
                "router_tau_scale": float(getattr(self, "_last_router_tau_scale", float("nan"))),
                "router_jitter_style": str(router_jitter_style),
                "router_jitter_style_resolved": str(router_jitter_style_resolved),
                "router_jitter_std": float(router_jitter_std),
                "router_jitter_enabled": int(router_jitter_enabled),
                "router_jitter_injected": int(router_jitter_injected),
                "router_jitter_target_flip_probability": float(router_jitter_target_flip_probability),
                "sigma_mean": float(router_jitter_sigma_mean),
                "sigma_std": float(router_jitter_sigma_std),
                "flip_probability_estimate": float(router_jitter_flip_probability_estimate),
                "router_entropy_reg_loss": float(router_entropy_reg_loss.detach().item()),
                "router_entropy_reg_active_frac": float(router_entropy_reg_active_frac.item()),
                "sum_g_mean": float(sum_g_mean),
                "sum_g_p90": float(sum_g_p90),
                "g0_mean": float(g0_mean),
                "g0_p90": float(g0_p90),
                "alpha_mean": float(alpha_mean),
                "alpha_p90": float(alpha_p90),
                "multiscale_norm": str(self.multiscale_norm_requested),
                "amplitude_multiplier": float(self.wavelet_ctxscale_amplitude_multiplier),
                "amplitude_multiplier_override": int(
                    bool(self.wavelet_ctxscale_amplitude_multiplier_override)
                ),
                "multiscale_k": int(self.wavelet_ctxscale_k_total),
                "norm_scale_mean": (
                    float(norm_scale_sample.mean().item())
                    if norm_scale_sample.numel() > 0
                    else float("nan")
                ),
                "norm_scale_p50": float(norm_scale_q["p50"]),
                "norm_scale_p90": float(norm_scale_q["p90"]),
                "norm_keff_mean": (
                    float(norm_keff_sample.mean().item())
                    if norm_keff_sample.numel() > 0
                    else float("nan")
                ),
                "norm_keff_p50": float(norm_keff_q["p50"]),
                "norm_keff_p90": float(norm_keff_q["p90"]),
                "bias_pre_norm_rms": float(norm_pre_rms),
                "bias_post_norm_rms": float(norm_post_rms),
                "film_enabled": int(bool(enable_film)),
                "film_nf_s_raw": int(film_nf_flags["s_raw"]),
                "film_nf_t_raw": int(film_nf_flags["t_raw"]),
                "film_nf_scale": int(film_nf_flags["scale"]),
                "film_nf_shift": int(film_nf_flags["shift"]),
                "film_sat_s": float(sat_s),
                "film_sat_t": float(sat_t),
                "film_sraw_sample": sraw_sample,
                "film_traw_sample": traw_sample,
                "film_scale_sample": scale_sample,
                "film_shift_sample": shift_sample,
                "bias_sample": bias_sample,
                "eff_sample": eff_sample,
                "base_sample": base_sample,
            }
            self._last_ctxscale_monitor_payload = payload

        return logits_out.to(dtype=compute_dtype), payload

    def debug_ctxscale_shift_v0_sanity_check(
        self,
        *,
        T: int = 128,
        B: int = 2,
        enable_film: bool = True,
        device: Optional[torch.device] = None,
    ):
        dev = device
        if dev is None:
            try:
                dev = self.wavelet_logit_bias_a.device
            except Exception:
                dev = torch.device("cpu")
        T = max(8, int(T))
        B = max(1, int(B))
        H = int(self.num_heads)
        D = int(self.head_dim)
        compute_dtype = torch.float32

        q = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
        w = torch.randn(B, T, H, D, device=dev, dtype=torch.float32)
        m_raw = torch.randn(B, H, T, T, device=dev, dtype=torch.float32)
        M_used = torch.tril(m_raw)
        hidden_states = torch.randn(B, T, self.hidden_size, device=dev, dtype=torch.float32)
        E_base_raw = torch.randn(B, H, T, T, device=dev, dtype=torch.float32)

        if self.wavelet_logit_bias_a.grad is not None:
            self.wavelet_logit_bias_a.grad = None
        for p in self.wavelet_bias_film.parameters():
            if p.grad is not None:
                p.grad = None

        logits_out, payload = self._build_ctxscale_shift_logit_bias_v0(
            q=q,
            w=w,
            M_used=M_used,
            hidden_states=hidden_states,
            E_base_raw=E_base_raw,
            T=T,
            compute_dtype=compute_dtype,
            need_log=True,
            layer_idx=self.layer_idx,
            step=0,
            enable_film=bool(enable_film),
        )
        loss = logits_out.float().mean()
        loss.backward()

        gate_grad = self.wavelet_logit_bias_a.grad
        gate_grad_finite_ratio = float("nan")
        if gate_grad is not None:
            g = gate_grad.detach().float()
            gate_grad_finite_ratio = float(torch.isfinite(g).to(dtype=torch.float32).mean().item())
        film_grad_norm = 0.0
        for p in self.wavelet_bias_film.parameters():
            if p.grad is not None:
                gg = p.grad.detach().float()
                if torch.isfinite(gg).all():
                    film_grad_norm += float(gg.norm().item())

        out = {
            "logits_finite": bool(torch.isfinite(logits_out).all().item()),
            "bias_finite": bool(torch.isfinite(payload.get("bias_sample")).all().item())
            if payload is not None and payload.get("bias_sample") is not None and payload.get("bias_sample").numel() > 0
            else True,
            "gate_grad_finite_ratio": float(gate_grad_finite_ratio),
            "film_grad_norm": float(film_grad_norm),
        }
        if self.wavelet_logit_bias_debug_assert:
            if not out["logits_finite"]:
                raise FloatingPointError("ctxscale_shift_v0 sanity check: non-finite logits_out")
            if not out["bias_finite"]:
                raise FloatingPointError("ctxscale_shift_v0 sanity check: non-finite bias sample")
            if not math.isfinite(out["gate_grad_finite_ratio"]) or out["gate_grad_finite_ratio"] < 0.99:
                raise FloatingPointError("ctxscale_shift_v0 sanity check: non-finite gate grad")
            if bool(enable_film) and out["film_grad_norm"] <= 0.0:
                raise FloatingPointError("ctxscale_shift_v0 sanity check: film grad norm is zero")
        return out

    @staticmethod
    def _finite_float_stats(values) -> tuple[float, float]:
        vals = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if len(vals) == 0:
            return float("nan"), float("nan")
        t = torch.tensor(vals, dtype=torch.float32)
        mean_v = float(t.mean().item())
        std_v = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
        return mean_v, std_v

    def _update_router_global_stats(self, *, step: int, layer_idx: int, payload: dict):
        if not self.training:
            return
        cfg = getattr(self, "config", None)
        if cfg is None or payload is None:
            return
        try:
            step_i = int(step)
            layer_i = int(layer_idx)
        except Exception:
            return
        # Keep runtime cache on the module (not config) to avoid JSON serialization
        # failures when Trainer saves config at checkpoints.
        cache = getattr(self, "_router_issue3_global_stats_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_router_issue3_global_stats_cache", cache)
        rec = cache.get(step_i)
        if rec is None:
            rec = {
                "seen_layers": set(),
                "pi_entropy": [],
                "pi_top1": [],
                "pi_margin": [],
                "sigma": [],
                "flip_prob": [],
                "reg_loss": [],
                "reg_active_frac": [],
                "emitted": False,
            }
            cache[step_i] = rec
        if layer_i in rec["seen_layers"]:
            return
        rec["seen_layers"].add(layer_i)
        for key, dst in (
            ("pi_entropy_mean", "pi_entropy"),
            ("pi_top1_mean", "pi_top1"),
            ("pi_margin_mean", "pi_margin"),
            ("sigma_mean", "sigma"),
            ("flip_probability_estimate", "flip_prob"),
            ("router_entropy_reg_loss", "reg_loss"),
            ("router_entropy_reg_active_frac", "reg_active_frac"),
        ):
            try:
                v = float(payload.get(key, float("nan")))
            except Exception:
                v = float("nan")
            if math.isfinite(v):
                rec[dst].append(v)

        total_layers = self._infer_total_layers_from_config(cfg)
        if total_layers is not None and total_layers > 0:
            should_emit = (len(rec["seen_layers"]) >= int(total_layers)) or (layer_i == int(total_layers) - 1)
        else:
            should_emit = len(rec["seen_layers"]) >= 1
        if rec.get("emitted", False) or (not should_emit):
            return

        pi_entropy_mean, pi_entropy_std = self._finite_float_stats(rec["pi_entropy"])
        pi_top1_mean, _ = self._finite_float_stats(rec["pi_top1"])
        pi_margin_mean, _ = self._finite_float_stats(rec["pi_margin"])
        sigma_mean, sigma_std = self._finite_float_stats(rec["sigma"])
        flip_prob_est, _ = self._finite_float_stats(rec["flip_prob"])
        reg_loss_mean, _ = self._finite_float_stats(rec["reg_loss"])
        reg_active_frac_mean, _ = self._finite_float_stats(rec["reg_active_frac"])
        self._k1_emit_log(
            f"[wavelet router global stats] step={step_i} "
            f"pi_entropy_mean={pi_entropy_mean:.6e} pi_entropy_std={pi_entropy_std:.6e} "
            f"pi_top1_mean={pi_top1_mean:.6e} pi_margin_mean={pi_margin_mean:.6e} "
            f"sigma_mean={sigma_mean:.6e} sigma_std={sigma_std:.6e} "
            f"flip_probability_estimate={flip_prob_est:.6e} "
            f"router_entropy_reg_loss={reg_loss_mean:.6e} "
            f"router_entropy_reg_active_frac={reg_active_frac_mean:.6e} "
            f"layers_seen={len(rec['seen_layers'])} total_layers={int(total_layers) if total_layers is not None else -1}"
        )
        rec["emitted"] = True
        for old_step in [k for k in cache.keys() if int(k) < step_i - 4]:
            cache.pop(old_step, None)

    def _log_ctxscale_shift_v0_monitor(
        self,
        *,
        layer_idx: Optional[int],
        step: int,
        payload: dict,
        attn_probs: torch.Tensor,
    ):
        bias_stats = _monitor_flat_stats(payload.get("bias_sample"))
        eff_stats = _monitor_flat_stats(payload.get("eff_sample"))
        base_stats = _monitor_flat_stats(payload.get("base_sample"))
        attn_stats = _monitor_attn_prob_stats(
            attn_probs,
            max_queries=int(self.wavelet_logit_bias_log_sample_tokens),
            max_heads=int(self.wavelet_logit_bias_log_sample_heads),
        )
        lid = int(self.layer_idx if layer_idx is None else layer_idx)
        g_layer_raw = float(payload["g_layer_raw"].detach().float().item())
        g_layer_raw_used = float(payload["g_layer_raw_used"].detach().float().item())
        g_layer = float(payload["g_layer"].detach().float().item())
        eff_t = payload.get("eff_sample")
        base_t = payload.get("base_sample")
        if torch.is_tensor(eff_t) and torch.is_tensor(base_t) and eff_t.numel() > 0 and base_t.numel() > 0:
            eff_abs_mean = float(eff_t.detach().float().abs().mean().item())
            base_abs_mean = float(base_t.detach().float().abs().mean().item())
            denom_floor = max(base_abs_mean * 1e-2, 1e-6)
            n_take = int(min(eff_t.numel(), base_t.numel()))
            eff_flat = eff_t.detach().float().reshape(-1)[:n_take].abs()
            base_flat = base_t.detach().float().reshape(-1)[:n_take].abs()
            eff_over_base_absmean = eff_abs_mean / max(base_abs_mean, 1e-12)
            eff_over_base_elem_mean = float((eff_flat / base_flat.clamp_min(denom_floor)).mean().item())
        else:
            eff_abs_mean = float("nan")
            base_abs_mean = float("nan")
            eff_over_base_absmean = float("nan")
            eff_over_base_elem_mean = float("nan")
        film_sraw_stats = _monitor_scalar_stats(payload.get("film_sraw_sample"))
        film_traw_stats = _monitor_scalar_stats(payload.get("film_traw_sample"))
        film_scale_stats = _monitor_scalar_stats(payload.get("film_scale_sample"))
        film_shift_stats = _monitor_scalar_stats(payload.get("film_shift_sample"))
        g_bias_std_finite = math.isfinite(float(eff_stats["std"]))
        router_gate_parts = []
        for name in (
            "sum_g_mean",
            "sum_g_p90",
            "g0_mean",
            "g0_p90",
            "alpha_mean",
            "alpha_p90",
        ):
            value = float(payload.get(name, float("nan")))
            if math.isfinite(value):
                router_gate_parts.append(f"{name}={value:.6e}")
        router_gate_stats = (
            " ".join(router_gate_parts) + " | "
            if router_gate_parts
            else ""
        )
        jitter_stats = ""
        if int(payload.get("router_jitter_enabled", 0)):
            jitter_stats = (
                f"jitter_style={payload.get('router_jitter_style_resolved', 'na')} "
                f"jitter_std={payload.get('router_jitter_std', float('nan')):.6e} "
                f"jitter_inj={int(payload.get('router_jitter_injected', 0))} "
                f"jitter_target_flip_prob="
                f"{payload.get('router_jitter_target_flip_probability', float('nan')):.6e} "
                f"jitter_sigma_mean={payload.get('sigma_mean', float('nan')):.6e} "
                f"jitter_sigma_std={payload.get('sigma_std', float('nan')):.6e} "
                f"jitter_flip_prob={payload.get('flip_probability_estimate', float('nan')):.6e} | "
            )
        norm_stats = (
            f"norm_mode={payload.get('multiscale_norm', 'none')} "
            f"norm_K={int(payload.get('multiscale_k', 1))} "
            f"norm_scale_mean={payload.get('norm_scale_mean', float('nan')):.6e} "
            f"norm_scale_p50={payload.get('norm_scale_p50', float('nan')):.6e} "
            f"norm_scale_p90={payload.get('norm_scale_p90', float('nan')):.6e} "
            f"bias_pre_norm_rms={payload.get('bias_pre_norm_rms', float('nan')):.6e} "
            f"bias_post_norm_rms={payload.get('bias_post_norm_rms', float('nan')):.6e}"
        )
        norm_keff_mean = float(payload.get("norm_keff_mean", float("nan")))
        if math.isfinite(norm_keff_mean):
            norm_stats += (
                f" k_eff_mean={norm_keff_mean:.6e} "
                f"k_eff_p50={payload.get('norm_keff_p50', float('nan')):.6e} "
                f"k_eff_p90={payload.get('norm_keff_p90', float('nan')):.6e}"
            )
        norm_stats += " | "
        film_stats = ""
        if int(payload.get("film_enabled", 0)):
            film_stats = (
                f"film_nf={int(payload.get('film_nf_s_raw', 0))},"
                f"{int(payload.get('film_nf_t_raw', 0))},"
                f"{int(payload.get('film_nf_scale', 0))},"
                f"{int(payload.get('film_nf_shift', 0))} "
                f"film_sat_s={payload.get('film_sat_s', float('nan')):.6e} "
                f"film_sat_t={payload.get('film_sat_t', float('nan')):.6e} | "
                f"film_s_raw mean={film_sraw_stats['mean']:.6e} "
                f"p50={film_sraw_stats['p50']:.6e} p90={film_sraw_stats['p90']:.6e} | "
                f"film_t_raw mean={film_traw_stats['mean']:.6e} "
                f"p50={film_traw_stats['p50']:.6e} p90={film_traw_stats['p90']:.6e} | "
                f"film_scale mean={film_scale_stats['mean']:.6e} "
                f"p50={film_scale_stats['p50']:.6e} p90={film_scale_stats['p90']:.6e} | "
                f"film_shift mean={film_shift_stats['mean']:.6e} "
                f"p50={film_shift_stats['p50']:.6e} p90={film_shift_stats['p90']:.6e} | "
            )
        msg = (
            f"[wavelet ctxscale_shift_v0 stats] layer={lid} step={int(step)} "
            f"g_layer_raw={g_layer_raw:.6e} g_layer_raw_used={g_layer_raw_used:.6e} "
            f"sig={payload['sig']:.6e} g_layer={g_layer:.6e} "
            f"sat_low={int(payload['sat_low'])} sat_high={int(payload['sat_high'])} "
            f"sat_extreme={int(payload['sat_extreme'])} "
            f"grad_abs={payload['grad_abs']:.6e} grad_p50={payload.get('grad_abs_p50', float('nan')):.6e} "
            f"grad_p90={payload.get('grad_abs_p90', float('nan')):.6e} grad_max={payload.get('grad_abs_max', float('nan')):.6e} "
            f"grad_zero_ratio={payload.get('grad_zero_ratio', float('nan')):.6e} "
            f"grad_finite={payload.get('grad_finite_ratio', float('nan')):.6e} "
            f"grad_nf={int(payload.get('grad_nonfinite', -1))} grad_missing={int(payload.get('grad_missing', 1))} "
            f"grad_zero={int(payload['grad_zero'])} req_grad={int(payload.get('param_req_grad', 0))} "
            f"delta_a={payload['delta_a']:.6e} update_ratio={payload['update_ratio']:.6e} "
            f"locked={int(payload['locked'])} raw_clamped={int(payload.get('raw_clamped', 0))} "
            f"raw_clamp_abs={payload.get('raw_clamp_abs', float('nan')):.6e} "
            f"autofix={int(payload.get('autofix_active', 0))} "
            f"feat_mode={payload.get('feat_mode', 'na')} "
            f"wavelet_mode={payload.get('wavelet_mode', 'na')} "
            f"basis_ctrl={payload.get('basis_control', 'none')} "
            f"router_mode={payload.get('router_mode', 'softmax')} "
            f"router_norm_mode={payload.get('router_norm_mode', 'none')} "
            f"router_cosine={payload.get('router_cosine', 0)} "
            f"router_tau_null={payload.get('router_tau_null', float('nan')):.6e} "
            f"router_tau_scale={payload.get('router_tau_scale', float('nan')):.6e} "
            f"{jitter_stats}"
            f"{router_gate_stats}"
            f"{norm_stats}"
            f"pi_entropy mean={payload['pi_entropy_mean']:.6e} p50={payload['pi_entropy_p50']:.6e} "
            f"p90={payload['pi_entropy_p90']:.6e} | "
            f"pi_top1 mean={payload['pi_top1_mean']:.6e} p90={payload['pi_top1_p90']:.6e} "
            f"| pi_margin mean={payload.get('pi_margin_mean', float('nan')):.6e} "
            f"p50={payload.get('pi_margin_p50', float('nan')):.6e} "
            f"p90={payload.get('pi_margin_p90', float('nan')):.6e} "
            f"| router_entropy_reg_loss={payload.get('router_entropy_reg_loss', float('nan')):.6e} "
            f"router_entropy_reg_active_frac={payload.get('router_entropy_reg_active_frac', float('nan')):.6e} "
            f"null_mean={payload['pi_null_mean']:.6e} | "
            f"rho p50={payload['rho_p50']:.6e} p90={payload['rho_p90']:.6e} p99={payload['rho_p99']:.6e} | "
            f"beta p50={payload['beta_p50']:.6e} p90={payload['beta_p90']:.6e} "
            f"p99={payload['beta_p99']:.6e} clamp_frac={payload['beta_clamp_frac']:.6e} "
            f"{payload.get('beta_per_scale_mean_str', '')}| "
            f"far_only={int(payload.get('far_only', 0))} far_min_delta={int(payload.get('far_min_delta', 0))} "
            f"head_frac={payload.get('head_frac', 1.0):.6e} head_count={int(payload.get('head_count', 0))} | "
            f"gate_branch={payload.get('gate_branch', 'na')} "
            f"cfg_disable_layer_gate={int(payload.get('cfg_disable_layer_gate', 0))} "
            f"cfg_use_head_gate={int(payload.get('cfg_use_head_gate', 0))} "
            f"head_gate={int(payload.get('head_gate', 0))} "
            f"disable_layer_gate={int(payload.get('disable_layer_gate', 0))} | "
            f"bias mean={bias_stats['mean']:.6e} std={bias_stats['std']:.6e} abs_p99={bias_stats['abs_p99']:.6e} | "
            f"g_bias mean={eff_stats['mean']:.6e} std={eff_stats['std']:.6e} abs_p99={eff_stats['abs_p99']:.6e} "
            f"std_finite={int(g_bias_std_finite)} | "
            f"base mean={base_stats['mean']:.6e} std={base_stats['std']:.6e} abs_p99={base_stats['abs_p99']:.6e} | "
            f"delta_over_base_absmean={eff_over_base_absmean:.6e} "
            f"delta_over_base_elem_mean={eff_over_base_elem_mean:.6e} "
            f"delta_abs_mean={eff_abs_mean:.6e} base_abs_mean={base_abs_mean:.6e} | "
            f"{film_stats}"
            f"attn_entropy mean={attn_stats['entropy_mean']:.6e} p50={attn_stats['entropy_p50']:.6e} "
            f"p90={attn_stats['entropy_p90']:.6e} | "
            f"attn_top1 mean={attn_stats['top1_mean']:.6e} p50={attn_stats['top1_p50']:.6e} "
            f"p90={attn_stats['top1_p90']:.6e} | "
            f"attn_margin mean={attn_stats['margin_mean']:.6e} p50={attn_stats['margin_p50']:.6e} "
            f"p90={attn_stats['margin_p90']:.6e}"
        )
        self._k1_emit_log(msg)
        self._update_router_global_stats(step=int(step), layer_idx=lid, payload=payload)

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
        if not bool(getattr(self, "eval_attn_heatmap_keep_counter", False)):
            self._eval_attn_heatmap_export_count = 0

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
        hidden_states=None,
        input_ids=None,
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
        wavelet_mode = self._normalize_wavelet_mode(getattr(config, "wavelet_mode", self.wavelet_mode))
        rel_layer_enabled = self._rel_layer_enabled(layer_idx, config=config)
        rel_enabled = rel_layer_enabled and (
            wavelet_mode in ("logit_bias_ctxscale_shift_v0", "logit_bias_ctxscale_shift_v0_film", "mlp_bias_baseline_v0")
            or wavelet_dtt is not None
        )
        logit_bias_payload = None
        logit_bias_step = None
        ctxscale_shift_payload = None

        # --- baseline raw logits and M_base ---
        E_base_raw, M_base, strict_WK, A, lower_QK, correction = path_ut_base_raw(
            q, k, w, beta, compute_dtype=compute_dtype,
            drift_dampen_ltrain=int(getattr(self, "wavelet_pa_state_drift_dampen_ltrain", 0)),
            drift_dampen_alpha=float(getattr(self, "wavelet_pa_state_drift_dampen_alpha", 1.0)),
        )
        self._last_pa_raw_logits_unconditional = E_base_raw.detach().to(dtype=torch.float32)
        if self.wavelet_pa_debug_store_correction_terms:
            self._last_pa_lower_QK = lower_QK.detach().to(dtype=torch.float32)
            self._last_pa_correction = correction.detach().to(dtype=torch.float32)
        else:
            self._last_pa_lower_QK = None
            self._last_pa_correction = None
        # --- pick M_used for defining QH in wavelet branch ---
        if use_wavelet_fused_H and wavelet_dtt is not None:
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
        if rel_enabled and wavelet_mode == "logit_bias":
            # Exp-A logit-bias:
            # z = z_path + g_layer * B_hat(delta), delta=(key_pos-query_pos)=(n-m).
            b_hat, eff_bias, g_layer, causal_2d = self._build_logit_bias_term(
                wavelet_dtt=wavelet_dtt,
                T=T,
                future=future,
                device=q.device,
                compute_dtype=compute_dtype,
            )
            rel = None
            E_wav_raw = E_base_raw + eff_bias.view(1, 1, T, T)
            if self.training:
                should_log_bias, logit_bias_step = self._logit_bias_should_log(global_step=global_step, config=config)
                if should_log_bias:
                    logit_bias_payload = {
                        "g_layer": g_layer.detach(),
                        "b_hat": b_hat.detach(),
                        "eff_bias": eff_bias.detach(),
                        "causal_2d": causal_2d.detach(),
                    }
        elif rel_enabled and wavelet_mode in (
            "logit_bias_ctxscale_shift_v0",
            "logit_bias_ctxscale_shift_v0_film",
            "mlp_bias_baseline_v0",
        ):
            should_log_ctx = False
            if self.training:
                should_log_ctx, logit_bias_step = self._logit_bias_should_log(global_step=global_step, config=config)
            else:
                eval_log_once = self._as_bool(
                    getattr(config, "wavelet_ctxscale_eval_log_once", True), default=True
                )
                if eval_log_once:
                    should_log_ctx = not bool(getattr(self, "_ctxscale_eval_log_done", False))
                    _, logit_bias_step = self._logit_bias_should_log(global_step=global_step, config=config)
                    if should_log_ctx:
                        self._ctxscale_eval_log_done = True
                else:
                    should_log_ctx, logit_bias_step = self._logit_bias_should_log(
                        global_step=global_step, config=config
                    )
            step_for_ctx = global_step if global_step is not None else logit_bias_step
            rel = None
            E_wav_raw, ctxscale_shift_payload = self._build_ctxscale_shift_logit_bias_v0(
                q=q,
                w=w,
                M_used=M_used,
                hidden_states=hidden_states,
                E_base_raw=E_base_raw,
                T=T,
                compute_dtype=compute_dtype,
                need_log=bool(should_log_ctx),
                layer_idx=layer_idx,
                step=step_for_ctx,
                enable_film=bool(wavelet_mode == "logit_bias_ctxscale_shift_v0_film"),
                k = k if self.bias_type == "rotary" else None,
            )
            # PAT-254 Task 1: causal-subtraction ablation hook. Forward-only,
            # additive on top of the real bias -- subtracts lambda * delta
            # from the combined logits, where `delta` is an externally
            # supplied [T,T] matrix (e.g. lambda * D @ P_Psi, or a matched
            # DCT-aligned control) built offline from a prior forward pass'
            # captured A^Q-off/A^PA. No effect unless explicitly enabled.
            _sub_spec = getattr(self, "_ctxscale_subtract_spec", None)
            if isinstance(_sub_spec, dict) and bool(_sub_spec.get("enabled", False)):
                _delta = _sub_spec["delta"].to(device=E_wav_raw.device, dtype=E_wav_raw.dtype)
                _lam = float(_sub_spec.get("lambda", 1.0))
                if _delta.shape[-1] < T:
                    _pad = T - _delta.shape[-1]
                    _delta = torch.nn.functional.pad(_delta, (0, _pad, 0, _pad))
                elif _delta.shape[-1] > T:
                    _delta = _delta[:T, :T]
                E_wav_raw = E_wav_raw - _lam * _delta.view(1, 1, T, T)
        elif rel_enabled and wavelet_mode == "router_rel":
            if router1 is not None and router2 is not None:
                rel = self.wavelet_rel_from_M_scale_router(
                    q, w, M_used, wavelet_dtt,
                    compute_dtype=compute_dtype,
                    d_chunk=d_chunk,
                    layer_idx=layer_idx,
                    rel_selection=rel_selection,
                    gate1=router1,
                    gate2=router2,
                    config=config,
                    global_step=global_step,
                )
            else:
                raise ValueError("router1 and router2 must be provided when wavelet_mode='router_rel'")
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

            step_val = global_step
            if step_val is None and config is not None:
                step_val = getattr(config, "router_global_step", None)
            step_val = self._to_int_or_none(step_val)

            log_every = router_log_every
            if log_every is None and config is not None:
                log_every = getattr(config, "router_log_every", 500)
            log_every = self._to_int_or_none(log_every)

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

        if wavelet_mode == "off":
            self._wavelet_analysis_emit_pa_baseline(
                layer_idx=layer_idx,
                step=global_step,
                T=T,
                B=B,
            )

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

        # PAT-105: fixed-interval cumulative mask.
        # At trigger positions t where (t+1) % k == 0: use path logits (E_wav_raw).
        # At non-trigger positions: replace with standard QK^T logits.
        # This turns path attention into a hybrid-kernel ablation: path attention is
        # active only at fixed intervals k, 2k, 3k, ...; elsewhere standard attention.
        _path_fixed_interval = int(getattr(config, 'path_fixed_interval', 0)) if config is not None else 0
        if _path_fixed_interval > 0:
            _pos_ids = torch.arange(T, device=q.device)
            _trigger = ((_pos_ids + 1) % _path_fixed_interval == 0)  # [T] bool
            _trigger_mask = _trigger.view(1, 1, T, 1)  # [1,1,T,1], broadcasts over [B,H,T,T]
            # Standard unscaled dot-product logits (same scale as E_wav_raw — scale applied later)
            _E_std_raw = torch.einsum(
                "bihd,bjhd->bhij",
                q.to(compute_dtype),
                k.to(compute_dtype),
            )
            # Trigger rows → path logits; non-trigger rows → standard QK^T logits
            E_wav_raw = torch.where(_trigger_mask, E_wav_raw, _E_std_raw)

        # PAT-225 capture (analysis-only, no effect on forward output): the
        # existing _last_logits_pa_only/_last_logits_full sets inside
        # _build_ctxscale_shift_logit_bias_v0 only fire for
        # wavelet_mode=="logit_bias_ctxscale_shift_v0". E_base_raw/E_wav_raw
        # are finalized here regardless of which wavelet_mode branch ran
        # above, so capture unconditionally here too (harmless redundant
        # overwrite with the same values for the ctxscale_shift_v0 case).
        # These three are opt-OUT (default True) rather than opt-in, so every
        # existing caller keeps working unchanged -- but they're expensive at
        # large model/seq_len (full [B,H,T,T] fp32 per layer, retained until
        # the next forward call), enough to OOM a 48GB card on a 24-layer/
        # 16-head medium model at L=4096 on their own. A script that only
        # needs a cheap capture (e.g. _last_router_pi below) should set
        # module._capture_debug_tensors = False on each PaTHAttention module
        # before its forward pass to skip this block.
        if bool(getattr(self, "_capture_debug_tensors", True)):
            self._last_logits_pa_only = E_base_raw.detach().to(dtype=torch.float32)
            self._last_logits_full = E_wav_raw.detach().to(dtype=torch.float32)
            # Value vectors used in the final attention-output einsum below
            # (out_wav = einsum("b h i j, b j h d -> b i h d", P_wav, v)). v is
            # untouched between the top-of-function _match_heads(v, Hw) and that
            # einsum, so this is exactly the tensor the output actually uses --
            # needed to decompose Delta(output) into an attention-weight-shift
            # term vs a value-content-shift term across two checkpoints.
            self._last_value_vectors = v.detach().to(dtype=torch.float32)

        P_base = None
        heatmap_enabled = bool(getattr(self, "eval_attn_heatmap_enabled", False)) or bool(getattr(self, "_debug_enabled", False))
        need_base_softmax = bool((analyzer is not None and rel is not None) or ((not self.training) and heatmap_enabled))
        if need_base_softmax:
            E_base = E_base_raw * scale
            base_fill = causal_mask_fill_value(E_base.dtype)
            E_base = E_base.masked_fill(future, base_fill)
            P_base = torch.softmax(E_base, dim=-1)

        E_wav = E_wav_raw * scale
        wave_fill = causal_mask_fill_value(E_wav.dtype)
        E_wav = E_wav.masked_fill(future, wave_fill)
        self._log_attn_margin_distance(
            layer_idx=layer_idx,
            masked_logits=E_wav,
            global_step=global_step,
        )
        P_wav = self._apply_eval_attn_norm(masked_logits=E_wav, future=future, layer_idx=layer_idx)
        self._maybe_capture_entmax_attention_stats(probs=P_wav, future=future, layer_idx=layer_idx)
        only_rel_layers = bool(getattr(self, "eval_attn_heatmap_only_rel_layers", False))
        if wavelet_mode == "cond_film_v2" and self.wavelet_cond_film_v2 is not None:
            self._wavelet_condfilm_v2_last_attn_stats = _monitor_attn_prob_stats(
                P_wav,
                max_queries=int(self.wavelet_logit_bias_log_sample_tokens),
                max_heads=int(self.wavelet_logit_bias_log_sample_heads),
            )
        else:
            self._wavelet_condfilm_v2_last_attn_stats = None
        if logit_bias_payload is not None and logit_bias_step is not None:
            self._log_logit_bias_monitor(
                layer_idx=layer_idx,
                step=int(logit_bias_step),
                g_layer=logit_bias_payload["g_layer"],
                b_hat=logit_bias_payload["b_hat"],
                eff=logit_bias_payload["eff_bias"],
                causal_mask_2d=logit_bias_payload["causal_2d"],
                attn_probs=P_wav,
            )
        if ctxscale_shift_payload is not None and logit_bias_step is not None:
            self._log_ctxscale_shift_v0_monitor(
                layer_idx=layer_idx,
                step=int(logit_bias_step),
                payload=ctxscale_shift_payload,
                attn_probs=P_wav,
            )
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
        out_base = None
        if P_base is not None and analyzer is not None and rel is not None:
            out_base = torch.einsum("b h i j, b j h d -> b i h d", P_base, v.to(compute_dtype))
        out_wav = torch.einsum("b h i j, b j h d -> b i h d", P_wav, v.to(compute_dtype))
        if getattr(self, "_mask_heads", None):
            for h in self._mask_heads:
                out_wav[:, :, h, :].zero_()
        if P_base is not None and (not only_rel_layers or bool(rel_enabled)):
            self._export_eval_attn_heatmaps(
                layer_idx=layer_idx,
                p_base=P_base,
                p_wav=P_wav,
                logits_base=E_base,
                logits_wav=E_wav,
                q_repr=q,
                k_repr=k,
                global_step=global_step,
                wavelet_mode=wavelet_mode,
                has_wavelet=bool(wavelet_dtt is not None),
                rel_applied=bool(rel_enabled),
                out_base=out_base,
                out_wav=out_wav,
                input_ids=input_ids,
            )

        if analyzer is not None and rel is not None and P_base is not None:
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
        if bool(getattr(self, "eval_attn_pattern_debug_tensors", False)):
            self._last_pat82_qk_stage_debug = {
                "hidden_states_input": _tensor_debug_summary_json("hidden_states_input", hidden_states),
                "pre_norm_q_flat": _tensor_debug_summary_json("pre_norm_q_flat", q),
                "pre_norm_k_flat": _tensor_debug_summary_json("pre_norm_k_flat", k),
            }
        else:
            self._last_pat82_qk_stage_debug = None
        self._last_router_entropy_reg_loss = hidden_states.new_zeros([])
        self._last_router_entropy_reg_active_frac = hidden_states.new_zeros([])
        global_step = kwargs.get("global_step", getattr(self.config, "router_global_step", None))
        router1, router2 = None, None
        record_router1, record_router2 = None, None
        rel_use = self._rel_layer_enabled(self.layer_idx, self.config)
        wavelet_mode = self._normalize_wavelet_mode(getattr(self.config, "wavelet_mode", self.wavelet_mode))
        router_active = bool(self.config.wavelet_router) and bool(rel_use) and (wavelet_mode == "router_rel")
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
                    # jitter uses only target flip ratio (router_jitter_flip_ratio).
                    jitter_std_base = getattr(self.config, "router_jitter_flip_ratio", None)
                    if jitter_std_base is None:
                        jitter_std_base = getattr(self.config, "router_jitter_std", 0.0)

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
                    jitter_std = self._resolve_router_jitter_std(
                        jitter_std_base,
                        global_step=global_step,
                        max_steps=max_steps,
                    )
                    tau_change_step = getattr(self.config, "tau_change_step", 0)
                    tau_change = getattr(self.config, "tau_change", 2.0)
                    log_every = int(getattr(self.config, "router_log_every", 500))
                    if tau_change_step > 0 and global_step == tau_change_step:
                        self.tau = tau_change

                    router1_logits = self._add_gaussian_jitter(
                        router1_logits,
                        jitter_std,
                        router_name="router1",
                        global_step=global_step,
                        max_steps=max_steps,
                    )
                    router1 = torch.softmax(router1_logits / self.tau, dim=-1)  # [B,T,H,S]

                    if self.router2 is None:
                        router2 = None
                    else:
                        router2_logits = self.router2(hidden_states)       # [B,T,H*S]
                        router2_logits = router2_logits.view(B, T, H, S)   # [B,T,H,S]
                        router2_logits = self._add_gaussian_jitter(
                            router2_logits,
                            jitter_std,
                            router_name="router2",
                            global_step=global_step,
                            max_steps=max_steps,
                        )
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
        if bool(getattr(self, "eval_attn_pattern_debug_tensors", False)):
            if not isinstance(self._last_pat82_qk_stage_debug, dict):
                self._last_pat82_qk_stage_debug = {}
            self._last_pat82_qk_stage_debug["post_norm_q_flat"] = _tensor_debug_summary_json("post_norm_q_flat", q)
            self._last_pat82_qk_stage_debug["post_norm_k_flat"] = _tensor_debug_summary_json("post_norm_k_flat", k)
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
            if bool(getattr(self, "eval_attn_pattern_debug_tensors", False)):
                self._last_pat82_qk_stage_debug["post_rearrange_q"] = _tensor_debug_summary_json("post_rearrange_q", q)
                self._last_pat82_qk_stage_debug["post_rearrange_k"] = _tensor_debug_summary_json("post_rearrange_k", k)
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
            if self.config.ablate_switch:
                ablate = ablation_from_conflict_csv("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/top_conflict_heads.csv", topk=5, layer_whitelist=[0,6])
            else:
                ablate = None

            path_attn_impl = self._normalize_path_attn_impl(
                getattr(self.config, "path_attn_impl", self.path_attn_impl)
            )
            self.path_attn_impl = path_attn_impl
            if path_attn_impl == "triton":
                o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
            else:
                o = self.path_attention_with_wavelet_QH(
                    q=q, k=k, v=v,
                    w=w, beta=beta,
                    wavelet_dtt=wavelet_decay_table,
                    use_wavelet_fused_H=False,
                    d_chunk=8,
                    compute_dtype=torch.float32,
                    analyzer=analyzer,
                    layer_idx=self.layer_idx,
                    ablate=ablate,
                    rel_selection=self.config.rel_selection,
                    router1=router1 if router_active else None,
                    router2=router2 if router_active else None,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    config=self.config,
                    global_step=global_step,
                    router_log_every=getattr(self.config, "router_log_every", 500),
                )
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
            # PAT-244: removed the temp/spectral distillation branch (dead in every
            # PAT-244/PAT-225 checkpoint this session -- distill_in_which_layers=0
            # made `self.layer_idx < self.config.distill_in_which_layers` false for
            # every layer, so dis_loss was always q.sum()*0.0 regardless of
            # spectral_loss_coe/distill_teacher). Still genuinely used by older
            # scripts with distill_in_which_layers>0 (train_*_distil_in_layer*.sh);
            # removing it means those are no longer reproducible as-is.
            dis_loss = q.sum() * 0.0
            # Keep the regularizer path minimal and easy to rollback: add weak entropy-floor
            # term directly into auxiliary loss used by the existing training objective.
            if self.training:
                reg_loss = getattr(self, "_last_router_entropy_reg_loss", None)
                if torch.is_tensor(reg_loss):
                    dis_loss = dis_loss + reg_loss.to(dtype=dis_loss.dtype, device=dis_loss.device)
            o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
            o = self.o_proj(o.to(hidden_states.dtype))
            if wavelet_mode == "cond_film_v2" and self.wavelet_cond_film_v2 is not None:
                o = self.wavelet_cond_film_v2(
                    q_in=hidden_states,
                    attn_out=o,
                    w_ctx=None,
                )
                self.wavelet_cond_film_v2.set_attn_stats(self._wavelet_condfilm_v2_last_attn_stats)
                self.wavelet_cond_film_v2.log_if_needed(
                    step=global_step,
                    layer_idx=self.layer_idx,
                    logger_obj=getattr(self, "logger", None),
                )
            # In ctxscale modes there is no explicit router branch, but we still expose
            # scale-mixture probabilities as router-like tensors for downstream analysis.
            if record_router1 is None:
                ctx_prob = getattr(self, "_last_ctxscale_router_prob", None)
                if torch.is_tensor(ctx_prob) and ctx_prob.dim() == 4:
                    record_router1 = ctx_prob
                    record_router2 = ctx_prob
            # if analyzer is not None:
            return o, None, past_key_values, dis_loss, record_router1, record_router2
            # else:
                # return o, None, past_key_values, dis_loss, None, None

        # ========= 其它路径（mask!=None）：最小实现 =========
        if self.use_w_shortconv:
            w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_dim)
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

        path_attn_impl = self._normalize_path_attn_impl(
            getattr(self.config, "path_attn_impl", self.path_attn_impl)
        )
        self.path_attn_impl = path_attn_impl
        if path_attn_impl == "triton":
            o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
        else:
            o = self.path_attention_with_wavelet_QH(
                q=q, k=k, v=v,
                w=w, beta=beta,
                wavelet_dtt=wavelet_decay_table,
                use_wavelet_fused_H=False,
                d_chunk=8,
                compute_dtype=torch.float32,
                analyzer=analyzer,
                layer_idx=self.layer_idx,
                ablate=None,
                rel_selection=self.config.rel_selection,
                router1=router1 if router_active else None,
                router2=router2 if router_active else None,
                hidden_states=hidden_states,
                input_ids=input_ids,
                config=self.config,
                global_step=global_step,
                router_log_every=getattr(self.config, "router_log_every", 500),
            )
        o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
        o = self.o_proj(o.to(hidden_states.dtype))
        if wavelet_mode == "cond_film_v2" and self.wavelet_cond_film_v2 is not None:
            o = self.wavelet_cond_film_v2(
                q_in=hidden_states,
                attn_out=o,
                w_ctx=None,
            )
            self.wavelet_cond_film_v2.set_attn_stats(None)
            self.wavelet_cond_film_v2.log_if_needed(
                step=global_step,
                layer_idx=self.layer_idx,
                logger_obj=getattr(self, "logger", None),
            )
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
        o = self.o_proj(o.to(hidden_states.dtype))
        return o, None, past_key_values
