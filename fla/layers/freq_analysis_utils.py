import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm


# ======================
# 1. 计算频谱统计 & 相似度
# ======================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F

def future_mask(T: int, device):
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1).view(1, 1, T, T)
import torch
import torch.nn.functional as F

from pathlib import Path
import json
import csv
import torch
@torch.no_grad()
def _corr_lower_triangle(E, R, eps=1e-9):
    # E,R: [B,H,T,T]
    B,H,T,_ = E.shape
    tri = torch.tril_indices(T, T, offset=0, device=E.device)
    e = E[:, :, tri[0], tri[1]]  # [B,H,N]
    r = R[:, :, tri[0], tri[1]]  # [B,H,N]
    e = e - e.mean(dim=-1, keepdim=True)
    r = r - r.mean(dim=-1, keepdim=True)
    num = (e * r).mean(dim=-1)
    den = (e.pow(2).mean(dim=-1).sqrt() * r.pow(2).mean(dim=-1).sqrt()).clamp_min(eps)
    return (num / den).mean(dim=0)  # -> [H]
import os
import torch
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm

@dataclass
class RunningCorr:
    # sums over samples: x, y, x^2, y^2, x*y
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    sum_x2: torch.Tensor
    sum_y2: torch.Tensor
    sum_xy: torch.Tensor
    count: torch.Tensor

def _init_running(L, H, S, device):
    z_hs = torch.zeros((L, H, S), device=device, dtype=torch.float64)
    z_h  = torch.zeros((L, H),    device=device, dtype=torch.float64)
    cnt  = torch.zeros((L,),      device=device, dtype=torch.float64)
    return RunningCorr(
        sum_x=z_hs.clone(),
        sum_y=z_h.clone(),
        sum_x2=z_hs.clone(),
        sum_y2=z_h.clone(),
        sum_xy=z_hs.clone(),
        count=cnt.clone(),
    )

def _finalize_corr(rc: RunningCorr, eps=1e-12):
    # corr per (L,H,S)
    Ex  = rc.sum_x  / rc.count[:, None, None].clamp_min(1.0)
    Ey  = rc.sum_y  / rc.count[:, None].clamp_min(1.0)
    Ex2 = rc.sum_x2 / rc.count[:, None, None].clamp_min(1.0)
    Ey2 = rc.sum_y2 / rc.count[:, None].clamp_min(1.0)
    Exy = rc.sum_xy / rc.count[:, None, None].clamp_min(1.0)

    varx = (Ex2 - Ex * Ex).clamp_min(0.0)
    vary = (Ey2 - Ey * Ey).clamp_min(0.0)
    cov  = Exy - Ex * Ey[:, :, None]
    corr = cov / (torch.sqrt(varx * vary[:, :, None] + eps))
    return corr

def _finalize_std_ratio(rc: RunningCorr, eps=1e-12):
    Ex  = rc.sum_x  / rc.count[:, None, None].clamp_min(1.0)
    Ey  = rc.sum_y  / rc.count[:, None].clamp_min(1.0)
    Ex2 = rc.sum_x2 / rc.count[:, None, None].clamp_min(1.0)
    Ey2 = rc.sum_y2 / rc.count[:, None].clamp_min(1.0)

    stdx = torch.sqrt((Ex2 - Ex * Ex).clamp_min(0.0) + eps)
    stdy = torch.sqrt((Ey2 - Ey * Ey).clamp_min(0.0) + eps)
    ratio = stdx / stdy[:, :, None]
    return ratio
import os
import math
import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def _safe_log(x, eps=1e-12):
    return torch.log(x.clamp_min(eps))

def _entropy(p):
    # p: [..., S]
    return -(p * _safe_log(p)).sum(dim=-1)

def _js_divergence(p, q):
    # p,q: [..., S]
    m = 0.5 * (p + q)
    kl_pm = (p * (_safe_log(p) - _safe_log(m))).sum(dim=-1)
    kl_qm = (q * (_safe_log(q) - _safe_log(m))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

@dataclass
class RouterStats:
    # sums for mean gate per (L,H,S)
    sum_gate1: torch.Tensor
    sum_gate2: torch.Tensor
    # token counts per layer (scalar per layer)
    tok_count: torch.Tensor

    # entropy sums per (L,H)
    sum_ent1: torch.Tensor
    sum_ent2: torch.Tensor

    # top1 prob sums per (L,H)
    sum_top1p1: torch.Tensor
    sum_top1p2: torch.Tensor

    # top1 usage counts per (L,H,S)
    top1_cnt1: torch.Tensor
    top1_cnt2: torch.Tensor

    # router1 vs router2 divergence per (L,H)
    sum_jsd: torch.Tensor

    # histograms of max prob (global per layer, optional)
    # bins: [0..1], store counts per layer
    maxp_hist1: torch.Tensor
    maxp_hist2: torch.Tensor
    hist_bins: torch.Tensor

class RouterAnalyzer:
    """
    Collects router statistics across layers during a forward pass over a dataset.

    Expected router1/router2 shape: [B, T, H, S] (softmax already applied).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_scales: int,
        save_dir: str,
        hist_bins: int = 20,
        compute_dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.L = n_layers
        self.H = n_heads
        self.S = n_scales
        self.save_dir = save_dir
        self.dtype = compute_dtype
        self.device = device

        os.makedirs(save_dir, exist_ok=True)

        # histogram bin edges for max-prob
        edges = torch.linspace(0.0, 1.0, steps=hist_bins + 1, device=device, dtype=self.dtype)

        self.st = RouterStats(
            sum_gate1=torch.zeros((self.L, self.H, self.S), device=device, dtype=self.dtype),
            sum_gate2=torch.zeros((self.L, self.H, self.S), device=device, dtype=self.dtype),
            tok_count=torch.zeros((self.L,), device=device, dtype=self.dtype),

            sum_ent1=torch.zeros((self.L, self.H), device=device, dtype=self.dtype),
            sum_ent2=torch.zeros((self.L, self.H), device=device, dtype=self.dtype),

            sum_top1p1=torch.zeros((self.L, self.H), device=device, dtype=self.dtype),
            sum_top1p2=torch.zeros((self.L, self.H), device=device, dtype=self.dtype),

            top1_cnt1=torch.zeros((self.L, self.H, self.S), device=device, dtype=self.dtype),
            top1_cnt2=torch.zeros((self.L, self.H, self.S), device=device, dtype=self.dtype),

            sum_jsd=torch.zeros((self.L, self.H), device=device, dtype=self.dtype),

            maxp_hist1=torch.zeros((self.L, hist_bins), device=device, dtype=self.dtype),
            maxp_hist2=torch.zeros((self.L, hist_bins), device=device, dtype=self.dtype),
            hist_bins=edges,
        )

    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        router1: torch.Tensor,  # [B,T,H,S]
        router2: torch.Tensor,  # [B,T,H,S]
        attention_mask: torch.Tensor = None,  # [B,T] with 1 valid, 0 pad
    ):
        li = int(layer_idx)
        assert 0 <= li < self.L, f"layer_idx {li} out of range"
        assert router1.dim() == 4 and router2.dim() == 4
        B, T, H, S = router1.shape
        assert (H, S) == (self.H, self.S), f"Expected (H,S)=({self.H},{self.S}), got ({H},{S})"
        assert router2.shape == router1.shape

        # p = router1.to(self.dtype)
        # q = router2.to(self.dtype)
        # IMPORTANT: avoid copying full [B,T,H,S] tensors unnecessarily.
        # Keep routers in their current dtype (float32 in your setup).
        p = router1
        q = router2
        acc_dtype = self.dtype  # dtype for accumulators (float32 in your setup)
        if attention_mask is None:
            # all tokens valid
            # m = torch.ones((B, T), device=p.device, dtype=self.dtype)
            m = torch.ones((B, T), device=p.device, dtype=p.dtype)
        else:
            # m = attention_mask.to(self.dtype)  # [B,T]
            m = attention_mask.to(device=p.device, dtype=p.dtype)
            assert m.shape == (B, T)

        # expand mask to [B,T,1,1]
        m4 = m.unsqueeze(-1).unsqueeze(-1)

        # token count (per layer): sum valid tokens * H (we’ll keep per-head sums separately)
        tok = m.sum()  # scalar: number of valid tokens in this batch
        # self.st.tok_count[li] += tok
        self.st.tok_count[li] += tok.to(acc_dtype)
        # mean gate per head-scale: sum over (B,T)
        # self.st.sum_gate1[li] += (p * m4).sum(dim=(0, 1))  # [H,S]
        # self.st.sum_gate2[li] += (q * m4).sum(dim=(0, 1))
        self.st.sum_gate1[li] += (p * m4).sum(dim=(0, 1)).to(acc_dtype)  # [H,S]
        self.st.sum_gate2[li] += (q * m4).sum(dim=(0, 1)).to(acc_dtype)

        # entropy per head: [B,T,H]
        ent1 = _entropy(p) * m.unsqueeze(-1)  # [B,T,H]
        ent2 = _entropy(q) * m.unsqueeze(-1)
        self.st.sum_ent1[li] += ent1.sum(dim=(0, 1))  # [H]
        self.st.sum_ent2[li] += ent2.sum(dim=(0, 1))

        # top1 prob and top1 index per token-head: [B,T,H]
        top1p1, top1i1 = p.max(dim=-1)
        top1p2, top1i2 = q.max(dim=-1)

        self.st.sum_top1p1[li] += (top1p1 * m.unsqueeze(-1)).sum(dim=(0, 1))  # [H]
        self.st.sum_top1p2[li] += (top1p2 * m.unsqueeze(-1)).sum(dim=(0, 1))

        # top1 usage counts per (H,S)
        # loop over heads to avoid huge one-hot tensor
        valid = (m > 0.0)  # [B,T] bool
        for h in range(self.H):
            idx1 = top1i1[:, :, h][valid]  # [N]
            idx2 = top1i2[:, :, h][valid]
            if idx1.numel() > 0:
                c1 = torch.bincount(idx1, minlength=self.S).to(self.dtype)
                c2 = torch.bincount(idx2, minlength=self.S).to(self.dtype)
                self.st.top1_cnt1[li, h] += c1
                self.st.top1_cnt2[li, h] += c2

        # router1 vs router2 divergence per head: JSD
        jsd = _js_divergence(p, q) * m.unsqueeze(-1)  # [B,T,H]
        self.st.sum_jsd[li] += jsd.sum(dim=(0, 1))  # [H]

        # histogram of max prob (aggregated across heads+tokens for each layer)
        # Use flattened valid values
        maxp1 = top1p1[valid].reshape(-1)  # [N*H]
        maxp2 = top1p2[valid].reshape(-1)

        # bucketize into bins
        edges = self.st.hist_bins
        # bucketize returns [0..len(edges)] ; clamp to [0..bins-1]
        b1 = torch.bucketize(maxp1, edges, right=True) - 1
        b2 = torch.bucketize(maxp2, edges, right=True) - 1
        b1 = b1.clamp(0, edges.numel() - 2)
        b2 = b2.clamp(0, edges.numel() - 2)

        h1 = torch.bincount(b1, minlength=edges.numel() - 1).to(self.dtype)
        h2 = torch.bincount(b2, minlength=edges.numel() - 1).to(self.dtype)
        self.st.maxp_hist1[li] += h1
        self.st.maxp_hist2[li] += h2

    @torch.no_grad()
    def finalize(self):
        """
        Return a dict of numpy arrays with means/usage/etc.
        """
        tok = self.st.tok_count.clamp_min(1.0)  # [L]

        # mean gate per (L,H,S)
        mean_gate1 = (self.st.sum_gate1 / tok[:, None, None]).cpu().numpy()
        mean_gate2 = (self.st.sum_gate2 / tok[:, None, None]).cpu().numpy()

        # entropy mean per (L,H)
        mean_ent1 = (self.st.sum_ent1 / tok[:, None]).cpu().numpy()
        mean_ent2 = (self.st.sum_ent2 / tok[:, None]).cpu().numpy()

        # top1 prob mean
        mean_top1p1 = (self.st.sum_top1p1 / tok[:, None]).cpu().numpy()
        mean_top1p2 = (self.st.sum_top1p2 / tok[:, None]).cpu().numpy()

        # top1 usage frequency per (L,H,S)
        # normalize by token count per layer (per head)
        # each head sees tok tokens, so counts / tok
        top1_freq1 = (self.st.top1_cnt1 / tok[:, None, None]).cpu().numpy()
        top1_freq2 = (self.st.top1_cnt2 / tok[:, None, None]).cpu().numpy()

        # jsd mean per (L,H)
        mean_jsd = (self.st.sum_jsd / tok[:, None]).cpu().numpy()

        # maxp histogram per layer -> normalize to prob
        hist1 = self.st.maxp_hist1.cpu().numpy()
        hist2 = self.st.maxp_hist2.cpu().numpy()
        hist1 = hist1 / (hist1.sum(axis=1, keepdims=True) + 1e-12)
        hist2 = hist2 / (hist2.sum(axis=1, keepdims=True) + 1e-12)
        edges = self.st.hist_bins.cpu().numpy()

        return {
            "mean_gate1": mean_gate1,
            "mean_gate2": mean_gate2,
            "mean_ent1": mean_ent1,
            "mean_ent2": mean_ent2,
            "mean_top1p1": mean_top1p1,
            "mean_top1p2": mean_top1p2,
            "top1_freq1": top1_freq1,
            "top1_freq2": top1_freq2,
            "mean_jsd": mean_jsd,
            "maxp_hist1": hist1,
            "maxp_hist2": hist2,
            "hist_edges": edges,
            "tok_count_per_layer": self.st.tok_count.cpu().numpy(),
        }

    def save_npz(self, name="router_analysis.npz"):
        out = self.finalize()
        path = os.path.join(self.save_dir, name)
        np.savez_compressed(path, **out)
        return path

    def finalize_and_plot(self, prefix="router"):
        out = self.finalize()
        npz_path = self.save_npz(f"{prefix}_analysis.npz")

        # ---- plotting helpers ----
        def savefig(fname):
            p = os.path.join(self.save_dir, fname)
            plt.savefig(p, dpi=200, bbox_inches="tight")
            plt.close()
            return p

        paths = []

        mean_gate1 = out["mean_gate1"]  # [L,H,S]
        mean_gate2 = out["mean_gate2"]
        mean_ent1  = out["mean_ent1"]   # [L,H]
        mean_ent2  = out["mean_ent2"]
        mean_top1p1 = out["mean_top1p1"]
        mean_top1p2 = out["mean_top1p2"]
        top1_freq1 = out["top1_freq1"]  # [L,H,S]
        top1_freq2 = out["top1_freq2"]
        mean_jsd = out["mean_jsd"]
        hist1 = out["maxp_hist1"]       # [L,BINS]
        hist2 = out["maxp_hist2"]
        edges = out["hist_edges"]

        L, H, S = mean_gate1.shape
        xS = np.arange(S)
        xL = np.arange(L)

        # (1) Layer x Scale heatmap (avg over heads)
        for tag, arr in [("router1", mean_gate1), ("router2", mean_gate2)]:
            plt.figure(figsize=(8, 5))
            plt.title(f"{tag}: mean gate (avg over heads) — layer x scale")
            plt.xlabel("scale")
            plt.ylabel("layer")
            img = arr.mean(axis=1)  # [L,S]
            plt.imshow(img, aspect="auto")
            plt.colorbar(label="mean gate")
            plt.xticks(np.arange(S))
            paths.append(savefig(f"{prefix}_{tag}_layer_scale_heatmap.png"))

        # (2) Layer x Head heatmap: entropy
        for tag, arr in [("router1", mean_ent1), ("router2", mean_ent2)]:
            plt.figure(figsize=(9, 5))
            plt.title(f"{tag}: entropy H(g) — layer x head (lower => more selective)")
            plt.xlabel("head")
            plt.ylabel("layer")
            plt.imshow(arr, aspect="auto")
            plt.colorbar(label="entropy")
            plt.xticks(np.arange(H))
            paths.append(savefig(f"{prefix}_{tag}_entropy_layer_head.png"))

        # (3) Layer x Head heatmap: mean top1 probability
        for tag, arr in [("router1", mean_top1p1), ("router2", mean_top1p2)]:
            plt.figure(figsize=(9, 5))
            plt.title(f"{tag}: mean max_s g_s (top-1 prob) — layer x head")
            plt.xlabel("head")
            plt.ylabel("layer")
            plt.imshow(arr, aspect="auto")
            plt.colorbar(label="mean top1 prob")
            plt.xticks(np.arange(H))
            paths.append(savefig(f"{prefix}_{tag}_top1prob_layer_head.png"))

        # (4) Layer x Scale heatmap: top1 usage frequency (avg over heads)
        for tag, arr in [("router1", top1_freq1), ("router2", top1_freq2)]:
            plt.figure(figsize=(8, 5))
            plt.title(f"{tag}: top-1 selected scale frequency (avg over heads) — layer x scale")
            plt.xlabel("scale")
            plt.ylabel("layer")
            img = arr.mean(axis=1)  # [L,S]
            plt.imshow(img, aspect="auto")
            plt.colorbar(label="freq")
            plt.xticks(np.arange(S))
            paths.append(savefig(f"{prefix}_{tag}_top1freq_layer_scale.png"))

        # (5) Router1 vs Router2: JSD layer x head
        plt.figure(figsize=(9, 5))
        plt.title("JSD(router1 || router2) — layer x head (0 => identical)")
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.imshow(mean_jsd, aspect="auto")
        plt.colorbar(label="JSD")
        plt.xticks(np.arange(H))
        paths.append(savefig(f"{prefix}_router1_vs_router2_jsd_layer_head.png"))

        # (6) Histograms of max prob per layer (two panels as two heatmaps)
        for tag, hist in [("router1", hist1), ("router2", hist2)]:
            plt.figure(figsize=(10, 4))
            plt.title(f"{tag}: distribution of max gate prob per layer (rows=layer, cols=bin)")
            plt.xlabel("max-prob bin")
            plt.ylabel("layer")
            plt.imshow(hist, aspect="auto")
            plt.colorbar(label="prob")
            paths.append(savefig(f"{prefix}_{tag}_maxprob_hist_layer.png"))

        # (7) For each layer: head x scale heatmap (mean gate) — save a grid of files
        for li in tqdm(range(L), desc="Saving per-layer head×scale heatmaps"):
            for tag, arr in [("router1", mean_gate1), ("router2", mean_gate2)]:
                plt.figure(figsize=(7, 4))
                plt.title(f"{tag}: mean gate — layer {li} (head x scale)")
                plt.xlabel("scale")
                plt.ylabel("head")
                plt.imshow(arr[li], aspect="auto")
                plt.colorbar(label="mean gate")
                plt.xticks(np.arange(S))
                plt.yticks(np.arange(H))
                paths.append(savefig(f"{prefix}_{tag}_layer{li:02d}_head_scale.png"))

        return npz_path, paths

class ScaleWiseAnalyzer:
    """
    scale-group-wise stats when P[D,T,T] shares scale every d_chunk dims.
    """
    def __init__(self, n_layers, n_heads, head_dim, d_chunk=8, device="cpu", save_dir=None):
        assert head_dim % d_chunk == 0, f"head_dim={head_dim} must be divisible by d_chunk={d_chunk}"
        self.L = n_layers
        self.H = n_heads
        self.D = head_dim
        self.dc = d_chunk
        self.S = head_dim // d_chunk
        self.device = device
        self.save_dir = save_dir

        # running stats for rel1, rel2, rel
        self.rc_rel1 = _init_running(self.L, self.H, self.S, device=device)
        self.rc_rel2 = _init_running(self.L, self.H, self.S, device=device)
        self.rc_rel  = _init_running(self.L, self.H, self.S, device=device)

        # optional: snapshot coe
        self.coe_snapshots = []  # list of dicts

    @torch.no_grad()
    def update(self, layer_idx, E_base_raw, q, q_corr, P, coe=None, use_lower_tri=True):
        """
        E_base_raw: [B,H,T,T]
        q, q_corr : [B,T,H,D]
        P        : [D,T,T]
        """
        B, H, T, T2 = E_base_raw.shape
        assert H == self.H and T == T2
        assert q.shape == (B, T, H, self.D)
        assert q_corr.shape == (B, T, H, self.D)
        assert P.shape == (self.D, T, T)

        # mask: lower-tri by default
        if use_lower_tri:
            mask2d = torch.tril(torch.ones((T, T), device=E_base_raw.device, dtype=torch.bool))
        else:
            mask2d = torch.ones((T, T), device=E_base_raw.device, dtype=torch.bool)
        mask_flat = mask2d.reshape(T * T)

        # ---- build scale-group P^(s) : [S,T,T] ----
        P_s = P.view(self.S, self.dc, T, T).mean(dim=1)  # [S,T,T]

        # ---- build q^(s) : [B,T,H,S] by summing within chunk ----
        q_s      = q.view(B, T, H, self.S, self.dc).sum(dim=-1)      # [B,T,H,S]
        qcorr_s  = q_corr.view(B, T, H, self.S, self.dc).sum(dim=-1) # [B,T,H,S]

        # ---- rel1^(s), rel2^(s): [B,H,T,T,S] ----
        rel1_s = torch.einsum("b t h s, s t n -> b h t n s", q_s,     P_s)
        rel2_s = torch.einsum("b t h s, s t n -> b h t n s", qcorr_s, P_s)
        rel_s  = rel1_s - rel2_s

        # ---- flatten over (T,T) then apply mask ----
        # rel_*_flat: [B,H,T*T,S] -> masked: [B,H,N,S]
        rel1_flat = rel1_s.reshape(B, H, T * T, self.S)[:, :, mask_flat, :]
        rel2_flat = rel2_s.reshape(B, H, T * T, self.S)[:, :, mask_flat, :]
        rel_flat  = rel_s .reshape(B, H, T * T, self.S)[:, :, mask_flat, :]

        # y: [B,H,N]
        y_flat = E_base_raw.reshape(B, H, T * T)[:, :, mask_flat]

        # counts
        N = y_flat.shape[-1]
        cnt = float(B * N)
        li = layer_idx

        # accumulate sums (float64 for stability)
        y64 = y_flat.to(torch.float64)
        for name, x_flat, rc in [
            ("rel1", rel1_flat, self.rc_rel1),
            ("rel2", rel2_flat, self.rc_rel2),
            ("rel",  rel_flat,  self.rc_rel),
        ]:
            x64 = x_flat.to(torch.float64)
            # sum over (B,N) -> [H,S]
            sum_x  = x64.sum(dim=(0, 2))
            sum_x2 = (x64 * x64).sum(dim=(0, 2))
            sum_y  = y64.sum(dim=(0, 2))           # [H]
            sum_y2 = (y64 * y64).sum(dim=(0, 2))   # [H]
            sum_xy = (x64 * y64.unsqueeze(-1)).sum(dim=(0, 2))  # [H,S]

            rc.sum_x [li] += sum_x
            rc.sum_x2[li] += sum_x2
            rc.sum_y [li] += sum_y
            rc.sum_y2[li] += sum_y2
            rc.sum_xy[li] += sum_xy
            rc.count[li]  += cnt

        if coe is not None:
            # store lightweight snapshot
            self.coe_snapshots.append({
                "layer": int(li),
                "coe": coe.detach().float().cpu().numpy(),
            })

    @torch.no_grad()
    def finalize(self):
        out = {}
        out["corr_rel1"] = _finalize_corr(self.rc_rel1).cpu().float().numpy()
        out["corr_rel2"] = _finalize_corr(self.rc_rel2).cpu().float().numpy()
        out["corr_rel"]  = _finalize_corr(self.rc_rel ).cpu().float().numpy()

        out["ratio_std_rel1"] = _finalize_std_ratio(self.rc_rel1).cpu().float().numpy()
        out["ratio_std_rel2"] = _finalize_std_ratio(self.rc_rel2).cpu().float().numpy()
        out["ratio_std_rel"]  = _finalize_std_ratio(self.rc_rel ).cpu().float().numpy()

        out["counts_per_layer"] = self.rc_rel.count.cpu().numpy()
        out["d_chunk"] = self.dc
        out["n_scales"] = self.S
        out["coe_snapshots"] = self.coe_snapshots
        return out

    def save_npz(self, filename="scale_wise_analyzer.npz"):
        assert self.save_dir is not None, "set save_dir first"
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        out = self.finalize()
        # coe_snapshots is ragged -> save separately
        coe_path = os.path.join(self.save_dir, "coe_snapshots.npy")
        np.save(coe_path, np.array(out["coe_snapshots"], dtype=object), allow_pickle=True)
        out = {k: v for k, v in out.items() if k != "coe_snapshots"}
        np.savez(path, **out)
        return path, coe_path

import json, csv
from pathlib import Path
import torch

# 你已有的函数：返回 [H]
# def _corr_lower_triangle(E_base_raw, rel, eps): ...

class LayerAttentionAnalyzer:
    """
    轻量统计：每层累计均值（标量 + 距离分桶 + head/位置分段）。
    - 不保存 [B,H,T,T] 大矩阵
    - update 必须在 no_grad 下
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        eps: float = 1e-9,
        lag_bins=(16, 64, 256),
        track_head_bucket: bool = True,
        track_pos_segments: bool = True,
        pos_segments: int = 16,   # early/mid/late
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eps = eps
        self.lag_bins = tuple(lag_bins)
        self.track_head_bucket = track_head_bucket
        self.track_pos_segments = track_pos_segments
        self.pos_segments = pos_segments

        # cache: (T, device, lag_bins) -> bucket_id_flat (for full), and for seg rows
        self._bucket_cache = {}

        self.reset()

    def reset(self):
        L = self.num_layers
        H = self.num_heads
        nb = len(self.lag_bins) + 1
        S = self.pos_segments

        self.count = torch.zeros(L, dtype=torch.long)

        # -------- scalar per-layer --------
        self.ratio_std = torch.zeros(L)      # std(rel)/std(E_base_raw)
        self.ratio_std_demean = torch.zeros(L)  # std(rel_demean)/std(E_demean)
        self.kl_mean   = torch.zeros(L)      # KL(P_wav||P_base)
        self.dH_mean   = torch.zeros(L)      # H(P_wav)-H(P_base)
        self.eta_mean  = torch.zeros(L)      # ||deltaQ||/||q||
        self.flip_mean = torch.zeros(L)      # top1 flip rate

        # -------- lag bucket mass per-layer (avg over head/batch) --------
        self.mass_base = torch.zeros(L, nb)  # [L,nb]
        self.mass_wav  = torch.zeros(L, nb)

        # -------- per-head scalars (existing + new) --------
        self.ratio_std_h = torch.zeros(L, H)
        self.ratio_std_demean_h = torch.zeros(L, H)
        self.kl_h        = torch.zeros(L, H)
        self.eta_h       = torch.zeros(L, H)
        self.corr_h      = torch.zeros(L, H)
        self.flip_h      = torch.zeros(L, H)

        # -------- NEW: per-head lag bucket mass --------
        if self.track_head_bucket:
            self.mass_base_hb = torch.zeros(L, H, nb)   # [L,H,nb]
            self.mass_wav_hb  = torch.zeros(L, H, nb)

        # -------- NEW: position-segment lag bucket mass --------
        if self.track_pos_segments:
            # [L,S,nb]  (avg over head/batch within segment)
            self.mass_base_seg = torch.zeros(L, S, nb)
            self.mass_wav_seg  = torch.zeros(L, S, nb)

    # -------------------------
    # math utils
    # -------------------------
    @torch.no_grad()
    def _entropy(self, P: torch.Tensor):
        return -(P * torch.log(P.clamp_min(self.eps))).sum(dim=-1)

    @torch.no_grad()
    def _kl(self, P: torch.Tensor, Q: torch.Tensor):
        return (P * (torch.log(P.clamp_min(self.eps)) - torch.log(Q.clamp_min(self.eps)))).sum(dim=-1)

    # -------------------------
    # NEW: fast lag-bucket mass with scatter
    # -------------------------
    @torch.no_grad()
    def _get_bucket_ids(self, T: int, device):
        """
        returns:
          bucket_id_full_flat: [T*T] in [0..nb-1]
          bucket_id_rows_flat: list of length S, each is [q_len*T]
          row_indices: list of tensors for each segment (on device)
        """
        key = (T, str(device), self.lag_bins, self.pos_segments)
        if key in self._bucket_cache:
            return self._bucket_cache[key]

        idx = torch.arange(T, device=device)
        lag = (idx.view(T, 1) - idx.view(1, T)).clamp_min(0)  # [T,T], causal lag

        boundaries = torch.tensor(self.lag_bins, device=device, dtype=lag.dtype)  # [nb-1]
        # bucket_id in [0..nb-1]
        bucket_id = torch.bucketize(lag, boundaries, right=False)  # [T,T]
        bucket_id_full_flat = bucket_id.reshape(-1)  # [T*T]

        # build segments over query positions (rows)
        S = self.pos_segments
        row_indices = []
        bucket_id_rows_flat = []
        if self.track_pos_segments:
            # split rows into S contiguous chunks
            # e.g., S=3: [0, T//3), [T//3, 2T//3), [2T//3, T)
            cuts = [int(round(T * i / S)) for i in range(S + 1)]
            for s in range(S):
                a, b = cuts[s], cuts[s + 1]
                rows = torch.arange(a, b, device=device)
                row_indices.append(rows)
                bucket_id_rows_flat.append(bucket_id[rows, :].reshape(-1))  # [(b-a)*T]

        out = (bucket_id_full_flat, bucket_id_rows_flat, row_indices)
        self._bucket_cache[key] = out
        return out

    @torch.no_grad()
    def _lag_bucket_mass_fast(self, P: torch.Tensor, per_head: bool = False, rows: torch.Tensor | None = None):
        """
        P: [B,H,T,T] causal
        rows: optional query row indices (subset of [0..T-1]) to compute segment mass
        returns:
          - if per_head=False: [nb] on CPU
          - if per_head=True : [H,nb] on CPU
        """
        B, H, T, _ = P.shape
        nb = len(self.lag_bins) + 1

        bucket_full_flat, bucket_rows_flat_list, row_indices_list = self._get_bucket_ids(T, P.device)

        if rows is None:
            bucket_flat = bucket_full_flat                         # [T*T]
            P_sum = P.sum(dim=0)                                   # [H,T,T] sum over batch
            P_flat = P_sum.reshape(H, -1)                          # [H,T*T]
        else:
            # P[:, :, rows, :] -> [B,H,q,T]
            P_sum = P[:, :, rows, :].sum(dim=0)                    # [H,q,T]
            P_flat = P_sum.reshape(H, -1)                          # [H, q*T]
            # need matching bucket_flat for these rows
            # we can regenerate from cache by matching rows to one of the stored segments if possible,
            # but safest: compute directly from bucket_id matrix slice:
            # (cost is tiny vs attention)
            idx = torch.arange(T, device=P.device)
            lag = (rows.view(-1, 1) - idx.view(1, -1)).clamp_min(0)     # [q,T]
            boundaries = torch.tensor(self.lag_bins, device=P.device, dtype=lag.dtype)
            bucket_flat = torch.bucketize(lag, boundaries, right=False).reshape(-1)  # [q*T]

        # scatter_add into [H,nb]
        out = torch.zeros(H, nb, device=P.device, dtype=P_flat.dtype)
        out.scatter_add_(dim=1, index=bucket_flat.view(1, -1).expand(H, -1), src=P_flat)

        # divide by B to match your previous mean over batch
        out = out / float(B)  # [H,nb]

        if per_head:
            return out.detach().cpu()          # [H,nb]
        else:
            return out.mean(dim=0).detach().cpu()  # [nb]

    @torch.no_grad()
    def _top1_flip_rate(self, P_base: torch.Tensor, P_wav: torch.Tensor):
        """
        returns:
          flip_mean: scalar
          flip_h: [H]
        """
        # [B,H,T]
        a = P_base.argmax(dim=-1)
        b = P_wav.argmax(dim=-1)
        flip = (a != b).float()
        flip_mean = flip.mean().item()
        flip_h = flip.mean(dim=(0, 2))  # [H]
        return flip_mean, flip_h.detach().cpu()

    # -------------------------
    # update
    # -------------------------
    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        E_base_raw: torch.Tensor,  # [B,H,T,T]
        rel: torch.Tensor,         # [B,H,T,T]
        P_base: torch.Tensor,      # [B,H,T,T]
        P_wav: torch.Tensor,       # [B,H,T,T]
        q: torch.Tensor,           # [B,T,H,d]
        deltaQ: torch.Tensor,      # [B,T,H,d]
    ):
        # ----- global scalars -----
        E = E_base_raw.float()
        R = rel.float()

        base_std = E.std()
        rel_std  = R.std()
        ratio = (rel_std / (base_std + self.eps)).item()

        # demean per-row (softmax-invariant direction removed)
        E_demean = E - E.mean(dim=-1, keepdim=True)
        R_demean = R - R.mean(dim=-1, keepdim=True)
        ratio_dm = (R_demean.std() / (E_demean.std() + self.eps)).item()

        KL = self._kl(P_wav, P_base).mean().item()

        Hb = self._entropy(P_base).mean().item()
        Hw = self._entropy(P_wav).mean().item()
        dH = (Hw - Hb)

        eta = (deltaQ.float().norm() / (q.float().norm() + self.eps)).item()

        flip_mean, flip_h = self._top1_flip_rate(P_base, P_wav)

        # lag bucket masses (avg over head/batch)
        mb = self._lag_bucket_mass_fast(P_base, per_head=False)  # [nb] CPU
        mw = self._lag_bucket_mass_fast(P_wav,  per_head=False)  # [nb] CPU

        # running average
        c = int(self.count[layer_idx].item())
        self.count[layer_idx] += 1
        c_new = c + 1

        def upd(old, val):
            return old + (val - old) / c_new

        self.ratio_std[layer_idx]        = upd(self.ratio_std[layer_idx], ratio)
        self.ratio_std_demean[layer_idx] = upd(self.ratio_std_demean[layer_idx], ratio_dm)
        self.kl_mean[layer_idx]          = upd(self.kl_mean[layer_idx], KL)
        self.dH_mean[layer_idx]          = upd(self.dH_mean[layer_idx], dH)
        self.eta_mean[layer_idx]         = upd(self.eta_mean[layer_idx], eta)
        self.flip_mean[layer_idx]        = upd(self.flip_mean[layer_idx], flip_mean)

        self.mass_base[layer_idx] = self.mass_base[layer_idx] + (mb - self.mass_base[layer_idx]) / c_new
        self.mass_wav[layer_idx]  = self.mass_wav[layer_idx]  + (mw - self.mass_wav[layer_idx])  / c_new

        # ----- per-head scalars -----
        base_std_h = E.std(dim=(0, 2, 3))                 # [H]
        rel_std_h  = R.std(dim=(0, 2, 3))                 # [H]
        ratio_h    = rel_std_h / (base_std_h + self.eps)  # [H]

        E_dm_h = E_demean.std(dim=(0, 2, 3))
        R_dm_h = R_demean.std(dim=(0, 2, 3))
        ratio_dm_h = R_dm_h / (E_dm_h + self.eps)         # [H]

        KL_full = self._kl(P_wav, P_base)                 # [B,H,T]
        kl_h = KL_full.mean(dim=(0, 2)).detach().cpu()    # [H]

        q_norm_h  = (q.float().pow(2).sum(dim=(0, 1, 3)) + self.eps).sqrt()       # [H]
        dQ_norm_h = (deltaQ.float().pow(2).sum(dim=(0, 1, 3)) + self.eps).sqrt()  # [H]
        eta_h = (dQ_norm_h / q_norm_h).detach().cpu()     # [H]

        corr_h = _corr_lower_triangle(E_base_raw, rel, eps=self.eps).detach().cpu()  # [H]

        self.ratio_std_h[layer_idx]        = self.ratio_std_h[layer_idx] + (ratio_h.detach().cpu()    - self.ratio_std_h[layer_idx]) / c_new
        self.ratio_std_demean_h[layer_idx] = self.ratio_std_demean_h[layer_idx] + (ratio_dm_h.detach().cpu() - self.ratio_std_demean_h[layer_idx]) / c_new
        self.kl_h[layer_idx]               = self.kl_h[layer_idx]        + (kl_h - self.kl_h[layer_idx]) / c_new
        self.eta_h[layer_idx]              = self.eta_h[layer_idx]       + (eta_h - self.eta_h[layer_idx]) / c_new
        self.corr_h[layer_idx]             = self.corr_h[layer_idx]      + (corr_h - self.corr_h[layer_idx]) / c_new
        self.flip_h[layer_idx]             = self.flip_h[layer_idx]      + (flip_h - self.flip_h[layer_idx]) / c_new

        # ----- NEW: per-head lag bucket mass -----
        if self.track_head_bucket:
            mb_hb = self._lag_bucket_mass_fast(P_base, per_head=True)  # [H,nb] CPU
            mw_hb = self._lag_bucket_mass_fast(P_wav,  per_head=True)  # [H,nb] CPU
            self.mass_base_hb[layer_idx] = self.mass_base_hb[layer_idx] + (mb_hb - self.mass_base_hb[layer_idx]) / c_new
            self.mass_wav_hb[layer_idx]  = self.mass_wav_hb[layer_idx]  + (mw_hb - self.mass_wav_hb[layer_idx])  / c_new

        # ----- NEW: position segments (early/mid/late) -----
        if self.track_pos_segments:
            B, H, T, _ = P_base.shape
            # segments based on query rows
            cuts = [int(round(T * i / self.pos_segments)) for i in range(self.pos_segments + 1)]
            for s in range(self.pos_segments):
                a, b = cuts[s], cuts[s + 1]
                if b <= a:
                    continue
                rows = torch.arange(a, b, device=P_base.device)
                mb_seg = self._lag_bucket_mass_fast(P_base, per_head=False, rows=rows)  # [nb] CPU
                mw_seg = self._lag_bucket_mass_fast(P_wav,  per_head=False, rows=rows)
                self.mass_base_seg[layer_idx, s] = self.mass_base_seg[layer_idx, s] + (mb_seg - self.mass_base_seg[layer_idx, s]) / c_new
                self.mass_wav_seg[layer_idx, s]  = self.mass_wav_seg[layer_idx, s]  + (mw_seg - self.mass_wav_seg[layer_idx, s])  / c_new

    # ---------- export ----------
    def as_dict(self):
        d = {
            "meta": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "eps": float(self.eps),
                "lag_bins": list(self.lag_bins),
                "track_head_bucket": bool(self.track_head_bucket),
                "track_pos_segments": bool(self.track_pos_segments),
                "pos_segments": int(self.pos_segments),
            },
            "count": self.count.cpu(),
            "ratio_std": self.ratio_std.cpu(),
            "ratio_std_demean": self.ratio_std_demean.cpu(),
            "kl_mean": self.kl_mean.cpu(),
            "dH_mean": self.dH_mean.cpu(),
            "eta_mean": self.eta_mean.cpu(),
            "flip_mean": self.flip_mean.cpu(),
            "mass_base": self.mass_base.cpu(),
            "mass_wav": self.mass_wav.cpu(),
            # head-level scalars
            "ratio_std_h": self.ratio_std_h.cpu(),
            "ratio_std_demean_h": self.ratio_std_demean_h.cpu(),
            "kl_h": self.kl_h.cpu(),
            "eta_h": self.eta_h.cpu(),
            "corr_h": self.corr_h.cpu(),
            "flip_h": self.flip_h.cpu(),
        }
        if self.track_head_bucket:
            d["mass_base_hb"] = self.mass_base_hb.cpu()  # [L,H,nb]
            d["mass_wav_hb"]  = self.mass_wav_hb.cpu()
        if self.track_pos_segments:
            d["mass_base_seg"] = self.mass_base_seg.cpu()  # [L,S,nb]
            d["mass_wav_seg"]  = self.mass_wav_seg.cpu()
        return d

    def save(self, out_dir: str, tag: str = "", step: int | None = None, make_plots: bool = True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        suffix = ""
        if tag:
            suffix += f"_{tag}"
        if step is not None:
            suffix += f"_step{step}"

        d = self.as_dict()

        torch.save(d, out / f"stats{suffix}.pt")

        # json (keep it readable)
        json_obj = {
            "meta": d["meta"],
            "count": d["count"].tolist(),
            "ratio_std": d["ratio_std"].tolist(),
            "ratio_std_demean": d["ratio_std_demean"].tolist(),
            "kl_mean": d["kl_mean"].tolist(),
            "dH_mean": d["dH_mean"].tolist(),
            "eta_mean": d["eta_mean"].tolist(),
            "flip_mean": d["flip_mean"].tolist(),
            "mass_base": d["mass_base"].tolist(),
            "mass_wav": d["mass_wav"].tolist(),
            "ratio_std_h": d["ratio_std_h"].tolist(),
            "ratio_std_demean_h": d["ratio_std_demean_h"].tolist(),
            "kl_h": d["kl_h"].tolist(),
            "eta_h": d["eta_h"].tolist(),
            "corr_h": d["corr_h"].tolist(),
            "flip_h": d["flip_h"].tolist(),
        }
        if "mass_base_hb" in d:
            json_obj["mass_base_hb"] = d["mass_base_hb"].tolist()
            json_obj["mass_wav_hb"]  = d["mass_wav_hb"].tolist()
        if "mass_base_seg" in d:
            json_obj["mass_base_seg"] = d["mass_base_seg"].tolist()
            json_obj["mass_wav_seg"]  = d["mass_wav_seg"].tolist()

        with open(out / f"stats{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2)

        # per-layer csv
        edges = (0,) + tuple(self.lag_bins) + ("inf",)
        bucket_names = [f"lag[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]

        csv_path = out / f"per_layer{suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["layer", "count", "ratio_std", "ratio_std_demean", "kl_mean", "dH_mean", "eta_mean", "flip_mean"]
            header += [f"mass_base_{bn}" for bn in bucket_names]
            header += [f"mass_wav_{bn}" for bn in bucket_names]
            writer.writerow(header)

            L = self.num_layers
            for l in range(L):
                row = [
                    l,
                    int(d["count"][l].item()),
                    float(d["ratio_std"][l].item()),
                    float(d["ratio_std_demean"][l].item()),
                    float(d["kl_mean"][l].item()),
                    float(d["dH_mean"][l].item()),
                    float(d["eta_mean"][l].item()),
                    float(d["flip_mean"][l].item()),
                ]
                row += [float(x) for x in d["mass_base"][l].tolist()]
                row += [float(x) for x in d["mass_wav"][l].tolist()]
                writer.writerow(row)

        # per-layer-head csv
        csv_path2 = out / f"per_layer_head{suffix}.csv"
        with open(csv_path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "head", "ratio_std_h", "ratio_std_demean_h", "kl_h", "eta_h", "corr_h", "flip_h"])
            L, H = d["ratio_std_h"].shape
            for l in range(L):
                for h in range(H):
                    writer.writerow([
                        l, h,
                        float(d["ratio_std_h"][l, h].item()),
                        float(d["ratio_std_demean_h"][l, h].item()),
                        float(d["kl_h"][l, h].item()),
                        float(d["eta_h"][l, h].item()),
                        float(d["corr_h"][l, h].item()),
                        float(d["flip_h"][l, h].item()),
                    ])

        # optional: per-layer-head-bucket csv
        if "mass_base_hb" in d:
            csv_path3 = out / f"per_layer_head_bucket{suffix}.csv"
            with open(csv_path3, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["layer", "head"] + [f"mass_base_{bn}" for bn in bucket_names] + [f"mass_wav_{bn}" for bn in bucket_names]
                writer.writerow(header)
                L, H, nb = d["mass_base_hb"].shape
                for l in range(L):
                    for h in range(H):
                        row = [l, h] + [float(x) for x in d["mass_base_hb"][l, h].tolist()] + [float(x) for x in d["mass_wav_hb"][l, h].tolist()]
                        writer.writerow(row)

        if make_plots:
            self._save_plots(out / f"figures{suffix}")

    def _save_plots(self, fig_dir: Path):
        fig_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        import numpy as np

        x = torch.arange(self.num_layers).cpu().numpy()

        def plot_line(y_tensor, title, ylabel, fname):
            y = y_tensor.cpu().numpy()
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("layer")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fig_dir / fname)
            plt.close()

        plot_line(self.ratio_std, "Wavelet strength ratio std(rel)/std(E_base_raw)", "ratio", "ratio_std.png")
        plot_line(self.ratio_std_demean, "Demeaned ratio std(rel-mean_row)/std(E-mean_row)", "ratio", "ratio_std_demean.png")
        plot_line(self.kl_mean, "KL(P_wav || P_base) per layer", "KL", "kl_mean.png")
        plot_line(self.dH_mean, "Delta entropy H(P_wav)-H(P_base) per layer", "dH", "dH_mean.png")
        plot_line(self.eta_mean, "eta = ||deltaQ||/||q|| per layer", "eta", "eta_mean.png")
        plot_line(self.flip_mean, "Top1 flip rate per layer (argmax change)", "flip", "flip_mean.png")

        # lag bucket masses
        edges = (0,) + self.lag_bins + (999999,)
        bucket_labels = [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges)-1)]
        nb = len(bucket_labels)

        def plot_bucket(mat, title, fname):
            plt.figure()
            for b in range(nb):
                plt.plot(x, mat[:, b].cpu().numpy(), label=bucket_labels[b])
            plt.xlabel("layer")
            plt.ylabel("mass")
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / fname)
            plt.close()

        plot_bucket(self.mass_base, "Lag-bucket attention mass (base)", "lag_bucket_mass_base.png")
        plot_bucket(self.mass_wav, "Lag-bucket attention mass (wavelet)", "lag_bucket_mass_wav.png")
        plot_bucket(self.mass_wav - self.mass_base, "Lag-bucket attention mass delta (wav-base)", "lag_bucket_mass_delta.png")

        # heatmaps helper
        def heatmap(mat, title, fname, xlabel="head", ylabel="layer"):
            M = mat.cpu().numpy()
            plt.figure()
            plt.imshow(M, aspect="auto", origin="lower")
            plt.colorbar()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_dir / fname)
            plt.close()

        heatmap(self.ratio_std_h, "ratio_std per (layer, head)", "ratio_std_h_heatmap.png")
        heatmap(self.ratio_std_demean_h, "ratio_std_demean per (layer, head)", "ratio_std_demean_h_heatmap.png")
        heatmap(self.kl_h, "KL(P_wav||P_base) per (layer, head)", "kl_h_heatmap.png")
        heatmap(self.eta_h, "eta per (layer, head)", "eta_h_heatmap.png")
        heatmap(self.corr_h, "corr(rel, E_base_raw) lower-tri per (layer, head)", "corr_h_heatmap.png")
        heatmap(self.flip_h, "top1 flip rate per (layer, head)", "flip_h_heatmap.png")

        # NEW: per-head bucket heatmaps
        if getattr(self, "track_head_bucket", False) and hasattr(self, "mass_base_hb"):
            for b in range(nb):
                heatmap(self.mass_base_hb[:, :, b], f"mass_base bucket {bucket_labels[b]} (layer x head)", f"mass_base_hb_bucket{b}.png")
                heatmap(self.mass_wav_hb[:, :, b],  f"mass_wav  bucket {bucket_labels[b]} (layer x head)", f"mass_wav_hb_bucket{b}.png")
                heatmap((self.mass_wav_hb - self.mass_base_hb)[:, :, b],
                        f"mass_delta bucket {bucket_labels[b]} (wav-base)", f"mass_delta_hb_bucket{b}.png")

        # NEW: position segments curves
        if getattr(self, "track_pos_segments", False) and hasattr(self, "mass_base_seg"):
            S = self.pos_segments
            for s in range(S):
                plot_bucket(self.mass_base_seg[:, s, :], f"Lag-bucket mass (base) segment {s}/{S}", f"lag_bucket_mass_base_seg{s}.png")
                plot_bucket(self.mass_wav_seg[:, s, :],  f"Lag-bucket mass (wav)  segment {s}/{S}", f"lag_bucket_mass_wav_seg{s}.png")
                plot_bucket((self.mass_wav_seg[:, s, :] - self.mass_base_seg[:, s, :]),
                            f"Lag-bucket mass delta (wav-base) segment {s}/{S}", f"lag_bucket_mass_delta_seg{s}.png")

import pandas as pd

def ablation_from_conflict_csv(path, topk=5, layer_whitelist=None):
    df = pd.read_csv(path).sort_values("conflict", ascending=False)
    if layer_whitelist is not None:
        df = df[df["layer"].isin(layer_whitelist)]
    ablate = {}
    for _, r in df.head(topk).iterrows():
        l, h = int(r["layer"]), int(r["head"])
        ablate.setdefault(l, []).append(h)
    return ablate


def path_ut_build_A(w, beta, compute_dtype=torch.float32):
    # w: [B,T,H,d], beta: [B,T,H]
    B, T, H, d = w.shape
    w0 = w.to(compute_dtype)
    b0 = beta.to(compute_dtype)
    eyeT = torch.eye(T, device=w.device, dtype=compute_dtype)
    w_scaled = w0 * b0.unsqueeze(-1)
    S = torch.einsum("b i h d, b j h d -> b h i j", w_scaled, w0)  # [B,H,T,T]
    A = torch.tril(S, diagonal=-1) + eyeT.view(1, 1, T, T)
    return A, S  # return S for monitoring L=strictLower(S)

def path_ut_M_from_S(A, S_lower, beta, compute_dtype=torch.float32):
    # A: [B,H,T,T] unit-lower, S_lower: [B,H,T,T] already tril(0), beta: [B,T,H]
    b0 = beta.to(compute_dtype)
    beta_h = b0.transpose(1, 2)  # [B,H,T]

    Zt = torch.linalg.solve_triangular(
        A.transpose(-1, -2),
        S_lower.transpose(-1, -2),
        upper=True,
        unitriangular=True,
    )
    Z = Zt.transpose(-1, -2)
    M = Z * beta_h.unsqueeze(2)  # right-multiply D=diag(beta)
    return M

def compute_path_wavelet_debug(
    q, k, v, w, beta, wavelet_dtt,
    compute_dtype=torch.float32,
    eps=1e-9,
):
    """
    Returns a dict with:
      E_base_raw, rel, E_wav_raw
      P_base, P_wav
      M_base, QH, deltaQ
      L_stats (from S = W D W^T)
    """
    B, T, H, d = q.shape
    scale = d ** -0.5
    fmask = future_mask(T, q.device)

    q0 = q.to(compute_dtype)
    k0 = k.to(compute_dtype)
    v0 = v.to(compute_dtype)
    w0 = w.to(compute_dtype)
    b0 = beta.to(compute_dtype)
    wav = wavelet_dtt.to(compute_dtype)

    # --- build A ---
    A, S_wdw = path_ut_build_A(w0, b0, compute_dtype=compute_dtype)
    L = torch.tril(S_wdw, diagonal=-1)
    L_max = L.abs().max()
    L_rms = (L.pow(2).mean()).sqrt()

    # --- base raw logits ---
    QK = torch.einsum("b i h d, b j h d -> b h i j", q0, k0)
    WK = torch.einsum("b i h d, b j h d -> b h i j", w0, k0)
    lower_QK  = torch.tril(QK, diagonal=0)
    strict_WK = torch.tril(WK, diagonal=-1)

    QW = torch.einsum("b i h d, b j h d -> b h i j", q0, w0)
    S_base = torch.tril(QW, diagonal=0)
    M_base = path_ut_M_from_S(A, S_base, b0, compute_dtype=compute_dtype)

    E_base_raw = lower_QK - (M_base @ strict_WK)  # NO scale, NO mask

    # --- QH and wavelet rel ---
    # deltaQ = M_base @ W,  QH = Q - deltaQ
    deltaQ = torch.einsum("b h i j, b j h d -> b i h d", M_base, w0)  # [B,T,H,d]
    QH = q0 - deltaQ

    # Wavelet PE term: rel = (QH) P^T
    rel = torch.einsum("b t h d, d t n -> b h t n", QH, wav)  # [B,H,T,T], NO scale, NO mask

    E_wav_raw = E_base_raw + rel

    # --- softmax distributions (for analysis) ---
    E_base = (E_base_raw * scale).masked_fill(fmask, float("-inf"))
    E_wav  = (E_wav_raw  * scale).masked_fill(fmask, float("-inf"))

    P_base = torch.softmax(E_base, dim=-1)
    P_wav  = torch.softmax(E_wav, dim=-1)

    out_base = torch.einsum("b h i j, b j h d -> b i h d", P_base, v0)
    out_wav  = torch.einsum("b h i j, b j h d -> b i h d", P_wav,  v0)

    return {
        "E_base_raw": E_base_raw,
        "rel": rel,
        "E_wav_raw": E_wav_raw,
        "P_base": P_base,
        "P_wav": P_wav,
        "M_base": M_base,
        "QH": QH,
        "deltaQ": deltaQ,
        "out_base": out_base,
        "out_wav": out_wav,
        "L_max": L_max,
        "L_rms": L_rms,
    }
import math
import torch

def mean_by_lag(mat_bhtt, max_lag=None):
    """
    mat: [B,H,T,T]
    returns: [H,L] where L=max_lag or T
    """
    B, H, T, _ = mat_bhtt.shape
    L = T if max_lag is None else min(max_lag, T)
    out = mat_bhtt.new_zeros((H, L))
    for lag in range(L):
        diag = mat_bhtt.diagonal(offset=-lag, dim1=-2, dim2=-1)  # [B,H,T-lag]
        out[:, lag] = diag.mean(dim=(0, 2))
    return out

def attention_entropy(P, eps=1e-9):
    # P: [B,H,T,T] -> entropy per query: [B,H,T]
    logP = torch.log(P.clamp_min(eps))
    return -(P * logP).sum(dim=-1)

def expected_lag(P):
    # P: [B,H,T,T] -> E[lag] per query: [B,H,T]
    B, H, T, _ = P.shape
    idx = torch.arange(T, device=P.device)
    lag = (idx.view(T, 1) - idx.view(1, T)).clamp_min(0).to(P.dtype)  # [T,T]
    return (P * lag.view(1, 1, T, T)).sum(dim=-1)

def kl_div(P, Q, eps=1e-9):
    # KL(P||Q) with P,Q: [B,H,T,T] -> [B,H,T]
    logP = torch.log(P.clamp_min(eps))
    logQ = torch.log(Q.clamp_min(eps))
    return (P * (logP - logQ)).sum(dim=-1)

def psd_from_lag_profile(profile_hl):
    # profile: [H,L] -> [H,L_fft]
    fft = torch.fft.rfft(profile_hl, dim=-1)
    return (fft.real**2 + fft.imag**2)

def analyze_debug(debug, max_lag=256, eps=1e-9, scale_groups=None):
    """
    debug: output of compute_path_wavelet_debug
    scale_groups: optional list[list[int]] to decompose wavelet dims by scale.
                 If provided, you must also provide wavelet table & q/QH outside,
                 or adapt this function accordingly. (see below example)
    """
    E_base_raw = debug["E_base_raw"]
    rel        = debug["rel"]
    P_base     = debug["P_base"]
    P_wav      = debug["P_wav"]
    out_base   = debug["out_base"]
    out_wav    = debug["out_wav"]
    deltaQ     = debug["deltaQ"]
    QH         = debug["QH"]

    # --- scalar norms ---
    # std ratios (global)
    base_std = E_base_raw.std()
    rel_std  = rel.std()
    ratio_std = rel_std / (base_std + 1e-12)

    # per-head std ratio
    base_std_h = E_base_raw.std(dim=(0, 2, 3))  # [H]
    rel_std_h  = rel.std(dim=(0, 2, 3))         # [H]
    ratio_std_h = rel_std_h / (base_std_h + 1e-12)

    # correlation on lower triangle (per-head)
    B, H, T, _ = E_base_raw.shape
    tri = torch.tril_indices(T, T, offset=0, device=E_base_raw.device)
    Eb = E_base_raw[:, :, tri[0], tri[1]]  # [B,H,NN]
    R  = rel[:, :, tri[0], tri[1]]         # [B,H,NN]

    Eb = Eb - Eb.mean(dim=-1, keepdim=True)
    R  = R  - R.mean(dim=-1, keepdim=True)
    corr_h = (Eb * R).mean(dim=-1) / ((Eb.pow(2).mean(dim=-1).sqrt() * R.pow(2).mean(dim=-1).sqrt()) + 1e-12)
    corr_h = corr_h.mean(dim=0)  # average over batch -> [H]

    # --- distribution shift ---
    KL = kl_div(P_wav, P_base, eps=eps)     # [B,H,T]
    KL_mean = KL.mean()
    KL_h = KL.mean(dim=(0, 2))              # [H]

    H_base = attention_entropy(P_base, eps=eps)  # [B,H,T]
    H_wav  = attention_entropy(P_wav,  eps=eps)
    H_base_mean = H_base.mean()
    H_wav_mean  = H_wav.mean()
    dH = (H_wav - H_base).mean()
    dH_h = (H_wav - H_base).mean(dim=(0, 2))

    Neff_base = torch.exp(H_base).mean()
    Neff_wav  = torch.exp(H_wav).mean()

    # --- expected lag ---
    lag_base = expected_lag(P_base).mean()
    lag_wav  = expected_lag(P_wav).mean()

    # --- lag profiles ---
    lagP_base = mean_by_lag(P_base, max_lag=max_lag)   # [H,L]
    lagP_wav  = mean_by_lag(P_wav,  max_lag=max_lag)
    lagRel    = mean_by_lag(rel,    max_lag=max_lag)
    lagEb     = mean_by_lag(E_base_raw, max_lag=max_lag)

    # --- spectrum ---
    psd_base = psd_from_lag_profile(lagP_base)
    psd_wav  = psd_from_lag_profile(lagP_wav)

    # --- QH transform strength ---
    eta = deltaQ.norm() / (QH.norm() + 1e-12)  # ||MW|| / ||QH||
    cos_Q_QH = ( ( (QH + deltaQ) * QH ).sum() /
                ((QH + deltaQ).norm() * QH.norm() + 1e-12) )  # Q = QH+deltaQ

    # --- output change ---
    out_diff = (out_wav - out_base).pow(2).mean().sqrt()

    return {
        "scalars": {
            "rel_std": rel_std.item(),
            "base_std": base_std.item(),
            "ratio_std": ratio_std.item(),
            "KL_mean": KL_mean.item(),
            "H_base_mean": H_base_mean.item(),
            "H_wav_mean": H_wav_mean.item(),
            "dH_mean": dH.item(),
            "Neff_base": Neff_base.item(),
            "Neff_wav": Neff_wav.item(),
            "E_lag_base": lag_base.item(),
            "E_lag_wav": lag_wav.item(),
            "L_max": float(debug["L_max"]),
            "L_rms": float(debug["L_rms"]),
            "eta_deltaQ_over_QH": eta.item(),
            "cos_Q_QH": cos_Q_QH.item(),
            "out_diff_rms": out_diff.item(),
        },
        "per_head": {
            "ratio_std_h": ratio_std_h.detach().cpu(),
            "corr_rel_base_h": corr_h.detach().cpu(),
            "KL_h": KL_h.detach().cpu(),
            "dH_h": dH_h.detach().cpu(),
        },
        "lag_profiles": {
            "P_base": lagP_base.detach().cpu(),
            "P_wav":  lagP_wav.detach().cpu(),
            "rel":    lagRel.detach().cpu(),
            "E_base_raw": lagEb.detach().cpu(),
        },
        "spectra": {
            "psd_base": psd_base.detach().cpu(),
            "psd_wav":  psd_wav.detach().cpu(),
        }
    }
def make_scale_groups(d: int, s: int):
    assert d % s == 0
    g = d // s
    return [list(range(i*g, (i+1)*g)) for i in range(s)]
@torch.no_grad()
def rel_by_scale_groups(QH, wavelet_dtt, groups, max_lag=256):
    # QH: [B,T,H,d], wavelet: [d,T,T]
    out = []
    for idxs in groups:
        wav_g = wavelet_dtt[idxs]                 # [g,T,T]
        QH_g  = QH[..., idxs]                     # [B,T,H,g]
        rel_g = torch.einsum("b t h g, g t n -> b h t n", QH_g, wav_g)  # [B,H,T,T]
        out.append(mean_by_lag(rel_g, max_lag=max_lag).cpu())          # [H,L]
    # shape: [s, H, L]
    return torch.stack(out, dim=0)
import torch
import torch.nn.functional as F
from collections import defaultdict

def _merge_stats_list(stats_list):
    """
    stats_list: List[Dict[band -> Tensor[H,S]]]
    """
    out = {}
    if not stats_list:
        return out
    # 默认所有 dict 的 band 集合相同
    bands = stats_list[0].keys()
    for b in bands:
        mats = [s[b] for s in stats_list if b in s]  # [n, H, S]
        out[b] = torch.stack(mats, dim=0).mean(dim=0)  # [H,S]
    return out

class SpectralCosStatsRecorder:
    def __init__(self):
        # layer_idx -> List[band_stats_dict]
        self._per_layer = defaultdict(list)

    def update(
        self,
        layer_idx: int,
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        eps: float = 1e-8,
        zero_mean: bool = True,
        scale_group: int = 8,
    ):
        """
        在训练/分析时调用：
          - layer_idx: 当前层编号 (int)
          - student, teacher: 当前层传进 distill 的 logits / features
        """
        stats = spectral_cos_3bands_stats(
            student,
            teacher,
            eps=eps,
            zero_mean=zero_mean,
            scale_group=scale_group,
        )
        self._per_layer[layer_idx].append(stats)

    def get_layer_mean(self):
        """
        返回:
          { layer_idx: { 'low': Tensor[H,S], 'mid':..., 'high':... }, ... }
        每个 Tensor 是在所有 step 上平均后的 [H, num_scales] cosine。
        """
        return {
            layer: _merge_stats_list(lst)
            for layer, lst in self._per_layer.items()
        }
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def plot_spectral_cos_heatmaps(
    layer_band_stats: dict,
    out_dir: str,
    cmap: str = "viridis",
):
    """
    layer_band_stats: 由 recorder.get_layer_mean() 得到的结构:
      {
        layer_idx: {
          'low':  Tensor[H,S],
          'mid':  Tensor[H,S],
          'high': Tensor[H,S],
        },
        ...
      }

    会输出:
      - 每一层一张 figure (包含 3 个 band 的 heatmap)
      - 一张 overall_spectral_cos.png：在层上平均后的各 band heatmap
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    band_names = ["low", "mid", "high"]
    layers = sorted(layer_band_stats.keys())

    # 累计 overall 用
    overall = {b: None for b in band_names}

    for layer in tqdm(layers, desc="Plotting spectral heatmaps"):
        stats = layer_band_stats[layer]

        fig, axes = plt.subplots(
            1, len(band_names),
            figsize=(5 * len(band_names), 4),
            squeeze=False
        )

        for j, band in enumerate(band_names):
            if band not in stats:
                axes[0, j].set_visible(False)
                continue

            mat = stats[band].detach().cpu().numpy()  # [H,S]

            # 累加到 overall
            if overall[band] is None:
                overall[band] = mat.copy()
            else:
                overall[band] += mat

            ax = axes[0, j]
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_title(f"Layer {layer} - {band}")
            ax.set_xlabel("scale group (D/8)")
            ax.set_ylabel("head")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_spectral_cos.png", dpi=150)
        plt.close(fig)

    # === overall: 在 layer 上平均 ===
    if layers:
        n_layers = len(layers)
        fig, axes = plt.subplots(
            1, len(band_names),
            figsize=(5 * len(band_names), 4),
            squeeze=False
        )

        for j, band in enumerate(band_names):
            ax = axes[0, j]
            if overall[band] is None:
                ax.set_visible(False)
                continue

            mat = overall[band] / float(n_layers)
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_title(f"Overall mean - {band}")
            ax.set_xlabel("scale group (D/8)")
            ax.set_ylabel("head")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(out_dir / "overall_spectral_cos.png", dpi=150)
        plt.close(fig)

def spectral_cos_3bands_stats(
    student: torch.Tensor,   # [B, L, H, D] or [B, Q, T, H, D]
    teacher: torch.Tensor,   # same shape as student
    *,
    eps: float = 1e-8,
    zero_mean: bool = True,
    scale_group: int = 8,
):
    """
    计算分 band 的 cos stats:
        返回 dict:
          {
            'low':  Tensor[H, S],  # S = D // scale_group
            'mid':  Tensor[H, S],
            'high': Tensor[H, S],
          }
    每个 entry 是在 batch / Q 上平均后的 [head, scale_group_index] 的 cosine similarity。
    """

    # 兼容 [B,L,H,D] 和 [B,Q,T,H,D]
    if student.dim() == 4:
        # [B,L,H,D] -> [B,Q=1,T=L,H,D]
        x_s = student.unsqueeze(1)
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
    assert D % scale_group == 0, f"D={D} 不能被 scale_group={scale_group} 整除"
    num_scales = D // scale_group

    # 2) 分 band
    seg = K // 3
    k1, k2 = seg, 2 * seg
    bands = {
        'low':  (0,  k1),
        'mid':  (k1, k2),
        'high': (k2, K),
    }

    stats = {}
    for name, (start, end) in bands.items():
        if end - start <= 1:
            # 频率数太少，跳过
            continue

        # 取出该 band: [B,Q,K_band,H,D]
        s_band = A_s[:, :, start:end]
        t_band = A_t[:, :, start:end]

        # 改成 [B,Q,H,D,K_band]
        s = s_band.permute(0, 1, 3, 4, 2).contiguous()
        t = t_band.permute(0, 1, 3, 4, 2).contiguous()

        if zero_mean:
            s = s - s.mean(dim=-1, keepdim=True)
            t = t - t.mean(dim=-1, keepdim=True)

        # 对 freq 维做 cosine_similarity:
        # s,t: [B,Q,H,D,K_band] → cos: [B,Q,H,D]
        cos = F.cosine_similarity(s, t, dim=-1, eps=eps)

        # 在 batch + Q 上平均 → [H,D]
        cos_h_d = cos.mean(dim=(0, 1))            # [H,D]

        # 按最后一维每 8 个 dim 分成一个 scale group
        cos_h_sg = cos_h_d.view(H, num_scales, scale_group)  # [H,S,8]
        cos_h_s = cos_h_sg.mean(dim=-1)          # [H,S]，每组 8 维取平均

        stats[name] = cos_h_s  # [H, num_scales]

    return stats

def spectrum_over_T_multi(x: torch.Tensor, eps: float = 1e-6):
    """
    x: [B, Q, T, H, D]
    沿 T 维 (dim=2) 做 rFFT，得到每个 (B,Q,H,D) 上的频谱。
    
    返回:
        A    : [B, Q, K, H, D]   幅值
        A_log: [B, Q, K, H, D]   log 幅值
        其中 K = T//2 + 1
    """
    assert x.dim() == 5, "x 应为 [B, Q, T, H, D]"
    B, Q, T, H, D = x.shape

    # rFFT over T dimension
    X = torch.fft.rfft(x, dim=2, norm='ortho')   # [B, Q, K, H, D]
    A = X.abs()
    # return A
    A_log = torch.log(A.clamp_min(eps))
    return A, A_log
# ========== 新增一个小工具函数：min-max 归一化 ==========
def min_max_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对 numpy 数组做 min-max 归一化到 [0, 1].
    如果所有值相等，会得到全 0.
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def analyze_teacher_student_groups(
    A_t_scale: torch.Tensor,   # [H,S,K]
    A_s_scale: torch.Tensor,   # [H,S,K]
    layer_idx: int,
    outdir_base: str = "plots_group_similarity",
    start_idx: int = 0,
):
    """
    对比 teacher / student 在 head_dim 分组后的频谱差异：
      1. 分别计算各自的 shape / energy 多样性指标
      2. 为 teacher / student 分别画:
         - group×group shape cos 相似度热力图
         - shape / energy 多样性条形图（min-max 归一化后）
         - 每个 head 的 group log 能量条形图（min-max 归一化后）
      3. 额外画一张 teacher vs student 的对比 summary 图（per-head，多样性值做联合 min-max）
    """

    os.makedirs(outdir_base, exist_ok=True)

    # 1) 分别计算统计量
    stats_t = compute_group_spectral_stats(A_t_scale)
    stats_s = compute_group_spectral_stats(A_s_scale)

    # 2) 逐个角色画图（带 tqdm）
    for name, stats in tqdm(
        [("teacher", stats_t), ("student", stats_s)],
        desc=f"layer{layer_idx} group spectrum plots"
    ):
        prefix = f"layer{layer_idx}_{name}_start_idx{start_idx}"
        plot_cos_shape_heatmaps(stats, title_prefix=prefix, outdir=outdir_base)
        plot_diversity_summary(stats, title_prefix=prefix, outdir=outdir_base)
        plot_group_energy(stats, title_prefix=prefix, outdir=outdir_base)

    # 3) 画一张 teacher vs student 的综合对比图（per-head）
    div_shape_t = stats_t["diversity_shape_per_head"].detach().cpu().numpy()   # [H]
    div_shape_s = stats_s["diversity_shape_per_head"].detach().cpu().numpy()   # [H]
    div_energy_t = stats_t["diversity_energy_per_head"].detach().cpu().numpy() # [H]
    div_energy_s = stats_s["diversity_energy_per_head"].detach().cpu().numpy() # [H]

    H = div_shape_t.shape[0]
    x = np.arange(H)
    width = 0.35

    # ---------- (a) shape 多样性对比：做 teacher+student 联合 min-max ----------
    shape_all = np.concatenate([div_shape_t, div_shape_s], axis=0)
    shape_all_norm = min_max_normalize_np(shape_all)
    div_shape_t_norm = shape_all_norm[:H]
    div_shape_s_norm = shape_all_norm[H:]

    plt.figure(figsize=(6, 3))
    plt.bar(x - width/2, div_shape_t_norm, width=width, label="teacher")
    plt.bar(x + width/2, div_shape_s_norm, width=width, label="student")
    plt.xlabel("head index")
    plt.ylabel("normalized shape diversity (min-max)")
    plt.title(
        f"Layer {layer_idx} – shape diversity per head\n"
        f"(raw mean T={stats_t['diversity_shape_mean']:.4f}, "
        f"S={stats_s['diversity_shape_mean']:.4f})"
    )
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(outdir_base, f"layer{layer_idx}_teacher_student_shape_diversity.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    # ---------- (b) energy 多样性对比：同样做联合 min-max ----------
    energy_all = np.concatenate([div_energy_t, div_energy_s], axis=0)
    energy_all_norm = min_max_normalize_np(energy_all)
    div_energy_t_norm = energy_all_norm[:H]
    div_energy_s_norm = energy_all_norm[H:]

    plt.figure(figsize=(6, 3))
    plt.bar(x - width/2, div_energy_t_norm, width=width, label="teacher")
    plt.bar(x + width/2, div_energy_s_norm, width=width, label="student")
    plt.xlabel("head index")
    plt.ylabel("normalized energy diversity (min-max)")
    plt.title(
        f"Layer {layer_idx} – energy diversity per head\n"
        f"(raw mean T={stats_t['diversity_energy_mean']:.4f}, "
        f"S={stats_s['diversity_energy_mean']:.4f})"
    )
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(outdir_base, f"layer{layer_idx}_teacher_student_energy_diversity.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    # 返回方便你后面 print / 做表（这里仍然是“原始值”）
    return stats_t, stats_s


def compute_group_spectral_stats(A_scale: torch.Tensor, eps: float = 1e-12):
    """
    A_scale: [H, S, K]  幅值 (非 log)，通常是你做完 rFFT 之后的 abs()，
             再按 dim 分组平均后的结果。

    返回:
      stats: dict 包含
        - P:           [H, S, K]   功率谱 |A|^2
        - E:           [H, S]      每个 head, group 的总能量
        - p:           [H, S, K]   归一化频谱 (sum_k p = 1)
        - cos_shape:   [H, S, S]   每个 head 内 group 之间的 shape 余弦相似度
        - diversity_shape_per_head: [H]   每个 head 的 shape 多样性
        - diversity_shape_mean: float     所有 head 平均 shape 多样性
        - diversity_energy_per_head: [H]  每个 head 的能量多样性 (Var log E)
        - diversity_energy_mean: float    所有 head 平均能量多样性
    """
    assert A_scale.dim() == 3, f"A_scale should be [H,S,K], got {A_scale.shape}"
    H, S, K = A_scale.shape

    # 功率谱
    P = (A_scale.clamp_min(0) ** 2)  # [H,S,K]

    # 总能量
    E = P.sum(dim=-1) + eps          # [H,S]

    # 归一化为频谱“形状”
    p = P / E.unsqueeze(-1)          # [H,S,K]

    # ---- shape 相似度：cos(p_i, p_j) ----
    p_norm = p / (p.norm(dim=-1, keepdim=True) + eps)  # [H,S,K]
    cos_shape = torch.einsum("hsk,hqk->hsq", p_norm, p_norm)  # [H,S,S]

    # 计算每个 head 的 shape 多样性: 1 - mean_offdiag(cos_shape)
    eye = torch.eye(S, device=A_scale.device, dtype=torch.bool)  # [S,S]
    off_diag = cos_shape[:, ~eye].view(H, -1)  # [H, S*(S-1)]
    diversity_shape_per_head = (1.0 - off_diag).mean(dim=-1)     # [H]
    diversity_shape_mean = diversity_shape_per_head.mean().item()

    # ---- 能量多样性：Var(log E) ----
    logE = E.log()  # [H,S]
    diversity_energy_per_head = logE.var(dim=-1)           # [H]
    diversity_energy_mean = diversity_energy_per_head.mean().item()

    stats = {
        "P": P,
        "E": E,
        "p": p,
        "cos_shape": cos_shape,
        "diversity_shape_per_head": diversity_shape_per_head,
        "diversity_shape_mean": diversity_shape_mean,
        "diversity_energy_per_head": diversity_energy_per_head,
        "diversity_energy_mean": diversity_energy_mean,
    }
    return stats


# ======================
# 2. 画图并保存
# ======================

def plot_cos_shape_heatmaps(stats, title_prefix: str, outdir: str = "plots_group_similarity"):
    """
    为每个 head 画一张 group×group 的 shape 余弦相似度热力图，并保存 PNG。
    """
    os.makedirs(outdir, exist_ok=True)
    cos_shape = stats["cos_shape"]  # [H,S,S]
    H, S, _ = cos_shape.shape

    for h in range(H):
        cs = cos_shape[h].detach().cpu().numpy()

        plt.figure(figsize=(4, 3))
        im = plt.imshow(cs, vmin=-1.0, vmax=1.0, origin="lower", aspect="equal", cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"{title_prefix} – head {h}\ncosine(shape) group×group")
        plt.xlabel("group index (s2)")
        plt.ylabel("group index (s1)")
        plt.tight_layout()
        fname = os.path.join(outdir, f"{title_prefix}_head{h}_cos_shape.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_diversity_summary(stats, title_prefix: str, outdir: str = "plots_group_similarity"):
    """
    画两张 summary 图：
      1. 每个 head 的 shape 多样性条形图（在本角色内做 min-max 归一化）
      2. 每个 head 的能量多样性条形图（在本角色内做 min-max 归一化）
    """
    os.makedirs(outdir, exist_ok=True)

    div_shape = stats["diversity_shape_per_head"].detach().cpu().numpy()   # [H]
    div_energy = stats["diversity_energy_per_head"].detach().cpu().numpy() # [H]
    H = div_shape.shape[0]
    x = np.arange(H)

    # 1) shape 多样性（本角色内部 min-max）
    div_shape_norm = min_max_normalize_np(div_shape)

    plt.figure(figsize=(6, 3))
    plt.bar(x, div_shape_norm)
    plt.xlabel("head index")
    plt.ylabel("normalized shape diversity (min-max)")
    plt.title(
        f"{title_prefix} – shape diversity per head\n"
        f"(raw mean={stats['diversity_shape_mean']:.4f})"
    )
    plt.tight_layout()
    fname = os.path.join(outdir, f"{title_prefix}_shape_diversity_bar.png")
    plt.savefig(fname, dpi=200)
    plt.close()

    # 2) 能量多样性（本角色内部 min-max）
    div_energy_norm = min_max_normalize_np(div_energy)

    plt.figure(figsize=(6, 3))
    plt.bar(x, div_energy_norm)
    plt.xlabel("head index")
    plt.ylabel("normalized energy diversity (min-max)")
    plt.title(
        f"{title_prefix} – energy diversity per head\n"
        f"(raw mean={stats['diversity_energy_mean']:.4f})"
    )
    plt.tight_layout()
    fname = os.path.join(outdir, f"{title_prefix}_energy_diversity_bar.png")
    plt.savefig(fname, dpi=200)
    plt.close()


def plot_group_energy(stats, title_prefix: str, outdir: str = "plots_group_similarity"):
    """
    画每个 head 的 group 能量分布（logE），
    在“同一个 head 的所有 group 上”做 min-max 归一化，让相对高低更明显。
    """
    os.makedirs(outdir, exist_ok=True)
    E = stats["E"]  # [H,S]
    H, S = E.shape
    logE = E.log().detach().cpu().numpy()   # [H,S]

    for h in range(H):
        vals = logE[h]                     # [S]
        vals_norm = min_max_normalize_np(vals)

        plt.figure(figsize=(4, 3))
        plt.bar(np.arange(S), vals_norm)
        plt.xlabel("group index (scale)")
        plt.ylabel("normalized log energy (min-max)")
        plt.title(f"{title_prefix} – head {h} log energy per group (normalized)")
        plt.tight_layout()
        fname = os.path.join(outdir, f"{title_prefix}_head{h}_logE_groups.png")
        plt.savefig(fname, dpi=200)
        plt.close()

# ======================
# 3. 使用示例
# ======================

# 假设你有 A_scale_no 和 A_scale_distill (shape=[H,S,K])
# A_scale_no = ...
# A_scale_distill = ...

# No-distill
# stats_no = compute_group_spectral_stats(A_scale_no)
# plot_cos_shape_heatmaps(stats_no, title_prefix="layer0_no_distill")
# plot_diversity_summary(stats_no, title_prefix="layer0_no_distill")
# plot_group_energy(stats_no, title_prefix="layer0_no_distill")

# Small-coeff distill
# stats_small = compute_group_spectral_stats(A_scale_distill)
# plot_cos_shape_heatmaps(stats_small, title_prefix="layer0_small_coeff")
# plot_diversity_summary(stats_small, title_prefix="layer0_small_coeff")
# plot_group_energy(stats_small, title_prefix="layer0_small_coeff")

# 假设你有 num_layers 个 stats dict
all_stats_per_layer = {}  # 长度 = num_layers，每个元素 = stats dict

# 训练/分析循环里 append
# all_stats_per_layer.append(stats_for_this_layer)
def plot_kl_cos_heatmap_over_layers(all_stats_per_layer, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    num_layers = len(all_stats_per_layer)

    # 假设 H,S 不变
    H, S = all_stats_per_layer[0]["kl_ts"].shape

    # [L, H]，对 scale 平均
    kl_matrix = torch.stack(
        [stats["kl_ts"].mean(dim=-1) for stats in all_stats_per_layer], dim=0
    )  # [L, H]
    cos_matrix = torch.stack(
        [stats["cos_sim"].mean(dim=-1) for stats in all_stats_per_layer], dim=0
    )  # [L, H]

    def plot_mat(mat, title, filename):
        data = mat.detach().cpu().numpy()
        plt.figure(figsize=(6, 5))
        im = plt.imshow(data, aspect="auto", origin="lower")
        plt.colorbar(im)
        plt.xlabel("head index")
        plt.ylabel("layer index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        plt.close()

    plot_mat(kl_matrix,  "KL(T||S) over layers & heads (mean over scale)",
             name+"kl_over_layers_heads.png")
    plot_mat(cos_matrix, "cos_sim(T,S) over layers & heads (mean over scale)",
             name+"cos_over_layers_heads.png")

def summarize_over_layers(all_stats_per_layer, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    num_layers = len(all_stats_per_layer)

    # 先把每层的 mean 值取出来
    def collect_metric(name):
        teacher_means = []
        student_means = []
        for stats in all_stats_per_layer:
            t = stats[f"{name}_t"]  # [H, S]
            s = stats[f"{name}_s"]
            # 对 head 和 scale 平均
            teacher_means.append(t.mean().item())
            student_means.append(s.mean().item())
        return np.array(teacher_means), np.array(student_means)

    r_low_t_mean, r_low_s_mean = collect_metric("r_low")
    centroid_t_mean, centroid_s_mean = collect_metric("centroid")
    entropy_t_mean, entropy_s_mean = collect_metric("entropy")

    layers = np.arange(num_layers)

    # 画三张图：r_low, centroid, entropy

    def plot_layer_line(x, y_t, y_s, title, ylabel, filename):
        plt.figure(figsize=(6, 4))
        plt.plot(x, y_t, marker="o", label="teacher")
        plt.plot(x, y_s, marker="o", label="student")
        plt.xlabel("layer index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        # 固定 y 轴范围到 [0, 1]
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        plt.close()

    plot_layer_line(layers, r_low_t_mean, r_low_s_mean,
                    "Low-frequency energy ratio over layers",
                    "r_low (mean over head,scale)",
                    name+"r_low_over_layers.png")

    plot_layer_line(layers, centroid_t_mean, centroid_s_mean,
                    "Spectral centroid over layers",
                    "centroid (mean over head,scale)",
                    name+"centroid_over_layers.png")

    plot_layer_line(layers, entropy_t_mean, entropy_s_mean,
                    "Spectral entropy over layers",
                    "entropy (mean over head,scale)",
                    name+"entropy_over_layers.png")

def compute_gamma_stats(
    w: torch.Tensor,      # [B, T, H, D]
    beta: torch.Tensor,   # [B, T, H]
    eps: float = 1e-12,
):
    """
    对 Householder-like 步骤 H_t = I - beta_t w_t w_t^T 做解析分析：

    在 w_t 方向上的特征值：
        lambda_t = 1 - gamma_t, 其中 gamma_t = beta_t * ||w_t||^2

    若 0 < gamma_t < 2 -> |lambda_t| < 1, 在 w_t 方向收缩；
    """
    assert w.dim() == 4 and beta.dim() == 3, "w:[B,T,H,D], beta:[B,T,H]"
    B, T, H, D = w.shape
    assert beta.shape == (B, T, H)

    # ||w_t||^2
    w_norm_sq = (w ** 2).sum(dim=-1)   # [B,T,H]

    gamma = beta * w_norm_sq           # [B,T,H]
    lambda_dir = 1.0 - gamma           # [B,T,H]

    # 在 w 方向是否收缩: |lambda| < 1
    is_contractive = (lambda_dir.abs() < 1.0)   # [B,T,H]

    ratio_contractive = is_contractive.float().mean().item()

    return {
        "gamma": gamma.detach(),                 # [B,T,H]
        "lambda_dir": lambda_dir.detach(),       # [B,T,H]
        "is_contractive": is_contractive.detach(),  # [B,T,H]
        "ratio_contractive": ratio_contractive,
        "num_heads": H,
    }
def plot_gamma_lambda_hist_by_head(
    gamma: torch.Tensor,          # [B,T,H]
    lambda_dir: torch.Tensor,     # [B,T,H]
    is_contractive: torch.Tensor, # [B,T,H] bool
    title_prefix: str,
    outdir: str = "plots_gamma_by_head",
    bins: int = 100,
):
    """
    按 head 分别画 gamma / lambda 的直方图。
    每个 head 会生成两张图片:
        {title_prefix}_head{h:02d}_gamma_hist.png
        {title_prefix}_head{h:02d}_lambda_hist.png
    """
    os.makedirs(outdir, exist_ok=True)

    assert gamma.shape == lambda_dir.shape == is_contractive.shape
    B, T, H = gamma.shape

    for h in range(H):
        # 取出单个 head: [B,T] → 展开成一维
        gamma_h = gamma[:, :, h].detach().cpu().numpy().ravel()
        lambda_h = lambda_dir[:, :, h].detach().cpu().numpy().ravel()
        contract_h = is_contractive[:, :, h].float().mean().item()

        # 去掉 NaN/Inf
        gamma_h = gamma_h[np.isfinite(gamma_h)]
        lambda_h = lambda_h[np.isfinite(lambda_h)]

        # ---- γ 直方图 ----
        plt.figure(figsize=(6, 4))
        plt.hist(gamma_h, bins=bins, density=True, alpha=0.8)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.axvline(2.0, color="k", linestyle="--", linewidth=1)
        plt.title(
            f"{title_prefix} – head {h} – gamma = beta * ||w||^2\n"
            f"ratio_contractive(|1-gamma|<1)={contract_h:.3f}"
        )
        plt.xlabel("gamma")
        plt.ylabel("density")
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"{title_prefix}_head{h:02d}_gamma_hist.png"),
            dpi=200,
        )
        plt.close()

        # ---- λ 直方图 ----
        plt.figure(figsize=(6, 4))
        plt.hist(lambda_h, bins=bins, density=True, alpha=0.8)
        plt.axvline(-1.0, color="k", linestyle="--", linewidth=1)
        plt.axvline(1.0,  color="k", linestyle="--", linewidth=1)
        plt.title(
            f"{title_prefix} – head {h} – lambda_dir = 1 - gamma"
        )
        plt.xlabel("lambda_dir (eigenvalue along w)")
        plt.ylabel("density")
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"{title_prefix}_head{h:02d}_lambda_hist.png"),
            dpi=200,
        )
        plt.close()

def compute_path_scores_and_xnorm_last_q(
    q: torch.Tensor,          # [B, H, D]   —— 使用最后一个 query（i = T-1）
    k: torch.Tensor,          # [B, T, H, D]
    w: torch.Tensor,          # [B, T, H, D]
    beta: torch.Tensor,       # [B, T, H]
    *,
    sqrt_d_scale: bool = False,   # 这里先不做 /sqrt(D)，分析形状
    show_progress: bool = False,
):
    """
    计算 Path Attention 下，最后一个 query（i=T-1）与所有 key(j) 的逐维匹配 scores，
    以及 path 变换后的 key 范数:
    
        x_j = ( ∏_{t=j}^{T-2} (I - β_{b,t,h} w_{b,t,h} w_{b,t,h}^T) ) k[b, j, h, :]
        
        scores[b, j, h, d] = q[b, h, d] * x_j[b, h, d]
        x_norm_sq[b, j, h] = ||x_j[b,h,:]||^2
    
    返回:
        scores:   [B, T, H, D]
        x_norm_sq:[B, T, H]
    复杂度: O(B·H·D·T²)，注意只在分析时用小 batch / 子序列。
    """

    assert q.dim() == 3,          "q 应为 [B,H,D]"
    assert k.dim() == 4,          "k 应为 [B,T,H,D]"
    assert w.dim() == 4,          "w 应为 [B,T,H,D]"
    assert beta.dim() == 3,       "beta 应为 [B,T,H]"

    B, H, D = q.shape
    assert k.size(0) == B and k.size(2) == H and k.size(3) == D
    assert w.size(0) == B and w.size(2) == H and w.size(3) == D
    assert beta.size(0) == B and beta.size(2) == H

    T = k.size(1)
    assert w.size(1) == T and beta.size(1) == T

    device = q.device
    dtype  = q.dtype

    scores = torch.empty((B, T, H, D), device=device, dtype=dtype)
    x_norm_sq = torch.empty((B, T, H), device=device, dtype=dtype)

    q_last = q[:, None, :, :]   # [B,1,H,D]
    j_iter = range(T)
    if show_progress:
        j_iter = tqdm(j_iter, desc="path_scores_last_q", leave=False)

    for j in j_iter:
        # 初始向量：每个 batch、每个 head 的 k_j
        x = k[:, j, :, :].clone()         # [B,H,D]

        if j < T - 1:
            for t in range(j, T - 1):
                w_t = w[:, t, :, :]                   # [B,H,D]
                b_t = beta[:, t, :].unsqueeze(-1)     # [B,H,1]
                dot = (x * w_t).sum(dim=-1, keepdim=True)  # [B,H,1]
                x = x - b_t * dot * w_t               # [B,H,D]

        # 范数平方: [B,H]
        x_norm_sq[:, j, :] = (x ** 2).sum(dim=-1)

        # 与最后 query 做逐维匹配
        s = q_last[:, 0, :, :] * x   # [B,H,D]
        if sqrt_d_scale:
            s = s / (D ** 0.5)
        scores[:, j, :, :] = s

    return scores, x_norm_sq
def plot_xnorm_decay(
    x_norm_sq: torch.Tensor,   # [B, T, H]
    setting_name: str,
    outdir: str = "plots_xnorm",
):
    """
    画出 E_{b,h}[||x_j||^2] 随 j (key position) 的变化曲线，
    用于观察路径长度越长，x_j 是否被更强地收缩。
    """
    os.makedirs(outdir, exist_ok=True)

    # [B,T,H] -> 平均到 [T]
    x_norm_mean = x_norm_sq.mean(dim=(0, 2))  # [T]
    x_norm_np = x_norm_mean.detach().cpu().numpy()

    T = x_norm_np.shape[0]
    js = np.arange(T)

    plt.figure(figsize=(6, 4))
    plt.plot(js, x_norm_np, linewidth=1.5)
    plt.xlabel("key index j (0...T-1)")
    plt.ylabel("E_{b,h}[||x_j||^2]")
    plt.title(f"{setting_name} – path-transformed key norm vs j")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{setting_name}_xnorm_decay.png"), dpi=200)
    plt.close()

def plot_layer_dashboard(stats, outdir, name):
    """
    stats: 上面那个 dict，所有 value shape 都是 [H, S]
    layer_idx: int，这一层的编号
    outdir: 保存目录
    """
    os.makedirs(outdir, exist_ok=True)

    # 转 numpy
    def to_np(x):
        return x.detach().cpu().numpy()

    r_low_t = to_np(stats["r_low_t"])      # [H, S]
    r_low_s = to_np(stats["r_low_s"])
    centroid_t = to_np(stats["centroid_t"])
    centroid_s = to_np(stats["centroid_s"])
    entropy_t = to_np(stats["entropy_t"])
    entropy_s = to_np(stats["entropy_s"])
    kl_ts = to_np(stats["kl_ts"])
    cos_sim = to_np(stats["cos_sim"])

    r_low_diff = r_low_s - r_low_t
    centroid_diff = centroid_s - centroid_t
    entropy_diff = entropy_s - entropy_t

    # 三行四列：第一行 teacher，第二行 student，第三行 Δ + cos_sim
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    fig.suptitle(f"{name} – spectral stats (head × scale)", fontsize=14)

    # helper
    def imshow(ax, data, title, cmap="viridis", center_zero=False, vmax=None, vmin=None):
        if vmax is not None and vmin is not None:
            pass
        elif center_zero:
            vmax = max(abs(data.min()), abs(data.max()))
            vmin = -vmax
        else:
            vmin, vmax = None, None
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("scale index")
        ax.set_ylabel("head index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 第一行：teacher 绝对值
    imshow(axes[0, 0], r_low_t,    "r_low (teacher)", vmax=1, vmin=0)
    imshow(axes[0, 1], centroid_t, "centroid (teacher)")
    imshow(axes[0, 2], entropy_t,  "entropy (teacher)")
    imshow(axes[0, 3], kl_ts,      "KL(T||S)", cmap="magma")

    # 第二行：student 绝对值
    imshow(axes[1, 0], r_low_s,    "r_low (student)", vmax=1, vmin=0)
    imshow(axes[1, 1], centroid_s, "centroid (student)")
    imshow(axes[1, 2], entropy_s,  "entropy (student)")
    imshow(axes[1, 3], kl_ts,      "KL(T||S)", cmap="magma")  # 这里仍然画 KL(T||S)，方便上下对比

    # 第三行：student vs teacher 差分 + cos_sim
    imshow(axes[2, 0], r_low_diff,    "Δ r_low (student - teacher)", center_zero=True, cmap="bwr")
    imshow(axes[2, 1], centroid_diff, "Δ centroid",                  center_zero=True, cmap="bwr")
    imshow(axes[2, 2], entropy_diff,  "Δ entropy",                   center_zero=True, cmap="bwr")
    imshow(axes[2, 3], cos_sim,       "cos_sim(T,S)",                cmap="viridis")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(outdir, f"{name}_dashboard.png")
    plt.savefig(fname, dpi=200)
    plt.close()


def spectrum_stats_teacher_student(A_t_scale, A_s_scale, low_ratio=0.25, eps=1e-12):
    """
    A_t_scale, A_s_scale: [H, S, K]
        H: head
        S: scale
        K: freq bins

    返回: 一个 dict，每个 key 的形状都是 [H, S]
    """
    assert A_t_scale.shape == A_s_scale.shape
    H, S, K = A_t_scale.shape

    # 这里把 A 当作幅度，平方成功率谱
    P_t = (A_t_scale.clamp_min(0)) ** 2  # [H, S, K]
    P_s = (A_s_scale.clamp_min(0)) ** 2

    # 归一化成“频谱分布”
    E_t = P_t.sum(dim=-1, keepdim=True) + eps
    E_s = P_s.sum(dim=-1, keepdim=True) + eps
    p_t = P_t / E_t
    p_s = P_s / E_s

    # 频率坐标
    freqs = torch.arange(K, device=A_t_scale.device).float()
    freqs = freqs.view(1, 1, K)  # 广播用

    # 低频能量比例
    cutoff = int(K * low_ratio)
    low_slice = slice(0, cutoff)

    r_low_t = p_t[..., low_slice].sum(dim=-1)   # [H, S]
    r_low_s = p_s[..., low_slice].sum(dim=-1)

    # 质心
    centroid_t = (p_t * freqs).sum(dim=-1)     # [H, S]
    centroid_s = (p_s * freqs).sum(dim=-1)

    # 频谱熵
    entropy_t = -(p_t * (p_t + eps).log()).sum(dim=-1)  # [H, S]
    entropy_s = -(p_s * (p_s + eps).log()).sum(dim=-1)

    # KL(T || S) on spectrum
    kl_ts = (p_t * ((p_t + eps).log() - (p_s + eps).log())).sum(dim=-1)  # [H, S]

    # 余弦相似度（用原始功率向量）
    num = (P_t * P_s).sum(dim=-1)   # [H, S]
    den = torch.sqrt((P_t ** 2).sum(dim=-1) + eps) * torch.sqrt((P_s ** 2).sum(dim=-1) + eps)
    cos_sim = num / (den + eps)

    return {
        "r_low_t": r_low_t,
        "r_low_s": r_low_s,
        "centroid_t": centroid_t,
        "centroid_s": centroid_s,
        "entropy_t": entropy_t,
        "entropy_s": entropy_s,
        "kl_ts": kl_ts,
        "cos_sim": cos_sim,
    }


def teacher_student_freq_distance(logits_T, logits_S, eps=1e-12):
    X_T = torch.fft.rfft(logits_T, dim=-1)
    X_S = torch.fft.rfft(logits_S, dim=-1)

    P_T = (X_T.abs() ** 2)
    P_S = (X_S.abs() ** 2)

    # L2 距离
    l2 = torch.sqrt(((P_T - P_S) ** 2).sum(dim=-1))  # [B, H]

    # KL on normalized spectra
    P_T_norm = P_T / (P_T.sum(dim=-1, keepdim=True) + eps)
    P_S_norm = P_S / (P_S.sum(dim=-1, keepdim=True) + eps)
    kl_TS = (P_T_norm * (P_T_norm + eps).log() - P_T_norm * (P_S_norm + eps).log()).sum(dim=-1)

    # cosine similarity
    num = (P_T * P_S).sum(dim=-1)
    den = torch.sqrt((P_T**2).sum(dim=-1) + eps) * torch.sqrt((P_S**2).sum(dim=-1) + eps)
    cos_sim = num / (den + eps)

    return {
        "l2": l2,
        "kl_TS": kl_TS,
        "cos_sim": cos_sim,
    }
def make_randomized_teacher_T(spectral_teacher_scores: torch.Tensor) -> torch.Tensor:
    """
    spectral_teacher_scores: [B, T, H, D]，已经做过沿 T 的 L2 norm。
    返回一个在 T 维上做了随机正交变换的 teacher，
    每个 (b,h,d) 的向量 L2 norm 不变，但 shape 被打乱。
    """
    B, T, H, D = spectral_teacher_scores.shape
    device = spectral_teacher_scores.device
    dtype = spectral_teacher_scores.dtype

    # 1) 生成 T x T 的随机正交矩阵 U_T
    U_T = random_orthogonal(T, device=device, dtype=dtype, det_one=True)  # [T, T]

    # 2) 把 [B, T, H, D] reshape 成 [B*H*D, T]，方便右乘 U_T^T
    x = spectral_teacher_scores.permute(0, 2, 3, 1).contiguous()  # [B, H, D, T]
    x = x.view(-1, T)                                             # [B*H*D, T]

    # 3) 在 T 维上做随机正交变换：x -> x @ U_T^T
    x_rand = x @ U_T.T                                            # [B*H*D, T]

    # 4) reshape 回原形状 [B, T, H, D]
    x_rand = x_rand.view(B, H, D, T).permute(0, 3, 1, 2).contiguous()  # [B,T,H,D]

    return x_rand

def random_orthogonal(
    dim: int,
    device=None,
    dtype=torch.float32,
    det_one: bool = True,
) -> torch.Tensor:
    """
    生成 dim x dim 的随机正交矩阵 U，满足 U^T U = I。
    如果 det_one=True，则额外保证 det(U) = 1（随机旋转矩阵）。
    """
    # 1. 随机高斯矩阵
    A = torch.randn(dim, dim, device=device, dtype=dtype)

    # 2. QR 分解得到 Q（正交）、R（上三角）
    #   torch.linalg.qr 对满秩方阵足够用
    Q, R = torch.linalg.qr(A)

    # 3. 为了数值稳定，一般把 R 的对角线符号吸收到 Q 里，让 R 的对角线非负
    diag_sign = torch.sign(torch.diag(R))
    # 避免 0 的情况
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign  # 广播到列上

    if det_one:
        # 4. 保证 det(Q) = 1
        det = torch.det(Q)
        if det < 0:
            # 翻转一列（或一行）就能把行列式从 -1 变成 +1
            Q[:, 0] = -Q[:, 0]

    return Q

def aggregate_spectrum_by_scale(
    A: torch.Tensor,
    group_size: int = 8,
    distill_teacher: str = "wavelet",  # "wavelet" or "rotary"
):
    """
    A: [B, Q, K, H, D]

    返回:
        A_scale: [H, S, K]
          H: head 数
          S: scale 数
             - wavelet: S = D // group_size (每 group_size 维一组求平均)
             - rotary : S = ceil(D / group_size) (每 group_size 取一个 dim: 0, group_size, 2*group_size, ...)
          K: 频率点数
    """
    assert A.dim() == 5, f"A 形状应为 [B, Q, K, H, D]，当前 {A.shape}"
    B, Q, K, H, D = A.shape

    # 先在 batch 和 query 上平均，减少噪声
    # [B, Q, K, H, D] -> [K, H, D]
    A_mean = A.mean(dim=(0, 1))  # [K, H, D]

    if distill_teacher == "wavelet":
        assert D % group_size == 0, "wavelet 模式下 D 必须能被 group_size 整除"
        S = D // group_size

        # [K, H, D] -> [H, D, K]
        A_mean = A_mean.permute(1, 2, 0).contiguous()  # [H, D, K]

        # 按 dim 维度分成 S 组，每组 group_size 维
        # [H, D, K] -> [H, S, group_size, K]
        A_group = A_mean.view(H, S, group_size, K)

        # 在 group_size 上平均，相当于同一 scale 的 group_size 维取平均
        # [H, S, group_size, K] -> [H, S, K]
        A_scale = A_group.mean(dim=2)

    elif distill_teacher in ["rotary", "shrink", "shrink_w_shuffle"]:
        assert group_size > 0, "group_size 必须为正整数"
        device = A_mean.device

        # 每 group_size 取一个维度: 0, group_size, 2*group_size, ...
        selected_indices = torch.arange(0, D, group_size, device=device)  # [S]
        S = selected_indices.numel()

        # [K, H, D] -> 选出这些 dim: [K, H, S]
        A_sel = A_mean[..., selected_indices]  # [K, H, S]

        # 调整为 [H, S, K]
        A_scale = A_sel.permute(1, 2, 0).contiguous()

    else:
        raise ValueError(f"未知的 distill_teacher='{distill_teacher}'，应为 'wavelet' 或 'rotary'")

    return A_scale


from scipy.optimize import linear_sum_assignment

import torch
from scipy.optimize import linear_sum_assignment  # 需要: pip install scipy

def match_heads_by_kl_over_S(
    A_t_scale: torch.Tensor,
    A_s_scale: torch.Tensor,
    eps: float = 1e-12,
):
    """
    A_t_scale, A_s_scale: [H, S, K]

    1) 在 S 维度上平均 -> [H, K]
    2) 在 K 维上归一化，视为频率分布
    3) 计算 KL(T_head_i || S_head_j) -> [H, H] 矩阵
    4) 用匈牙利算法做一一匹配，得到最小总 KL 的匹配关系

    返回:
        t_mean: [H, K] teacher 在 S 上平均后的频谱
        s_mean: [H, K] student 在 S 上平均后的频谱
        kl_mat: [H, H] KL 矩阵，kl_mat[i,j] = KL(T_i || S_j)
        row_ind: [H]，teacher head 索引
        col_ind: [H]，与之匹配的 student head 索引（匈牙利算法结果）
    """
    assert A_t_scale.shape == A_s_scale.shape, "teacher/student 形状必须一致"
    H, S, K = A_t_scale.shape

    # 1) 在 S 维上平均 -> [H, K]
    t_mean = A_t_scale.mean(dim=1)  # [H, K]
    s_mean = A_s_scale.mean(dim=1)  # [H, K]

    # 2) 视作频谱强度，归一化成在 K 维上的分布
    t_spec = t_mean.clamp_min(0)
    s_spec = s_mean.clamp_min(0)

    t_dist = t_spec / (t_spec.sum(dim=-1, keepdim=True) + eps)  # [H, K]
    s_dist = s_spec / (s_spec.sum(dim=-1, keepdim=True) + eps)  # [H, K]

    # 3) 计算 KL(T_i || S_j) 矩阵
    # p: [H, 1, K], q: [1, H, K]
    p = t_dist[:, None, :]  # teacher
    q = s_dist[None, :, :]  # student

    kl_mat = (p * ((p + eps).log() - (q + eps).log())).sum(dim=-1)  # [H, H]

    # 4) 匈牙利算法求一一匹配（最小化总 KL）
    cost = kl_mat.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)  # row: teacher, col: student

    # row_ind 默认是 0..H-1 顺序，但为了保险，你也可以按 row_ind 排序后再用
    return t_mean, s_mean, kl_mat, row_ind, col_ind


def plot_matched_heads_over_freq(
    t_mean: torch.Tensor,
    s_mean: torch.Tensor,
    row_ind,
    col_ind,
    save_path: str,
    title_prefix: str = "",
):
    """
    t_mean: [H, K] teacher 在 S 上平均后的频谱 (over scale)
    s_mean: [H, K] student 在 S 上平均后的频谱
    row_ind, col_ind: 匈牙利算法输出的匹配 (teacher_idx, student_idx)
    """
    H, K = t_mean.shape
    t_np = t_mean.detach().cpu().numpy()
    s_np = s_mean.detach().cpu().numpy()

    # 转成 numpy 数组（scipy 输出可能是 np.ndarray）
    row_ind = list(row_ind)
    col_ind = list(col_ind)

    x = list(range(K))

    n_pairs = len(row_ind)
    ncols = 4
    nrows = math.ceil(n_pairs / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True
    )
    axes = axes.flatten()

    for idx in range(n_pairs):
        t_h = int(row_ind[idx])  # teacher head index
        s_h = int(col_ind[idx])  # student head index
        ax = axes[idx]

        ax.plot(x, t_np[t_h], label=f"T head {t_h}")
        ax.plot(x, s_np[s_h], label=f"S head {s_h}", linestyle="--")

        ax.set_title(f"T{t_h} vs S{s_h}")
        ax.set_xlabel("freq index (K)")
        ax.set_ylabel("mean over scales (S)")
        ax.legend(fontsize=8)

    # 去掉多余子图
    for k in range(n_pairs, len(axes)):
        fig.delaxes(axes[k])

    if title_prefix:
        fig.suptitle(
            f"{title_prefix} – head spectral profiles (Hungarian KL matching)",
            fontsize=14
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92 if title_prefix else 1.0])
    plt.savefig(save_path, dpi=200)
    plt.close()
import os
import json
import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch


@dataclass
class TokenScaleDumperConfig:
    out_dir: str
    tag: str = ""
    compress: bool = True              # write .csv.gz if True else .csv
    max_rows_per_shard: int = 5_000_000  # rotate files to avoid huge single file
    flush_every: int = 50_000          # flush every N rows
    eps: float = 1e-9

    # what to record
    record_mu: bool = True
    record_lowmass: bool = True
    record_entropy: bool = True
    record_top1: bool = True

    # token position segmentation
    pos_segments: int = 16              # e.g. 3 -> early/mid/late

    # define "low-frequency" (s large => low-freq)
    # lowmass = sum_{s>=low_start} gate[..., s]
    low_frac: float = 0.25             # last 25% scales are "low-freq"

    # optional extra columns (if provided to update())
    record_surprisal: bool = True
    record_longmass: bool = True


class TokenScaleDumper:
    """
    Stream token-scale routing records to disk (no in-memory accumulation).

    Expected inputs:
      gate1, gate2: [B, T, H, S] (softmaxed already)
      input_ids:   [B, T] (int64)
    Optional:
      surprisal:   [B, T]  (float)
      longmass_base:  [B, H, T] (float)  # e.g. long bucket attention mass per token
      longmass_delta: [B, H, T] (float)  # wav-base longmass per token

    Each row corresponds to one (sample b, position t, head h) at a given layer.
    """

    def __init__(self, cfg: TokenScaleDumperConfig):
        self.cfg = cfg
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self._rows_written_total = 0
        self._rows_written_in_shard = 0
        self._shard_idx = 0

        self._f = None
        self._writer = None

        self._scale_idx_cache: Dict[int, torch.Tensor] = {}  # S -> [S] float32 on CPU
        self._header = None

        self._open_new_shard()

        # write meta once
        meta = {
            "type": "TokenScaleDumper",
            "tag": cfg.tag,
            "compress": cfg.compress,
            "max_rows_per_shard": cfg.max_rows_per_shard,
            "flush_every": cfg.flush_every,
            "pos_segments": cfg.pos_segments,
            "low_frac": cfg.low_frac,
            "record_mu": cfg.record_mu,
            "record_lowmass": cfg.record_lowmass,
            "record_entropy": cfg.record_entropy,
            "record_surprisal": cfg.record_surprisal,
            "record_longmass": cfg.record_longmass,
        }
        with open(self.out / f"token_scale_meta{self._suffix()}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _suffix(self) -> str:
        return f"_{self.cfg.tag}" if self.cfg.tag else ""

    def _shard_path(self, shard_idx: int) -> Path:
        ext = "csv.gz" if self.cfg.compress else "csv"
        return self.out / f"token_scale_records{self._suffix()}_shard{shard_idx:05d}.{ext}"

    def _open_new_shard(self):
        self.close(shutdown=False)

        path = self._shard_path(self._shard_idx)
        if self.cfg.compress:
            self._f = gzip.open(path, "at", encoding="utf-8", newline="")
        else:
            self._f = open(path, "a", encoding="utf-8", newline="")

        self._writer = csv.writer(self._f)
        self._rows_written_in_shard = 0

        # write header on each shard for convenience
        self._header = self._build_header()
        self._writer.writerow(self._header)
        self._f.flush()

    def _build_header(self):
        # minimal identifiers
        header = [
            "step", "layer", "b", "t", "T", "pos_seg", "h", "token_id",
        ]

        # gate1 / gate2 stats
        if self.cfg.record_top1:
            header += ["top1_s1", "top1p1", "top1_s2", "top1p2"]
        if self.cfg.record_entropy:
            header += ["ent1", "ent2"]
        if self.cfg.record_mu:
            header += ["mu1", "mu2"]
        if self.cfg.record_lowmass:
            header += ["lowmass1", "lowmass2"]

        # optional extras
        if self.cfg.record_surprisal:
            header += ["surprisal"]
        if self.cfg.record_longmass:
            header += ["longmass_base", "longmass_delta"]

        return header

    def _get_scale_idx_cpu(self, S: int) -> torch.Tensor:
        # cached CPU float tensor [S] used for mu computation
        if S not in self._scale_idx_cache:
            self._scale_idx_cache[S] = torch.arange(S, dtype=torch.float32, device="cpu")
        return self._scale_idx_cache[S]

    @torch.no_grad()
    def update(
        self,
        *,
        step: int,
        layer_idx: int,
        input_ids: torch.Tensor,          # [B,T]
        gate1: torch.Tensor,              # [B,T,H,S] softmaxed
        gate2: torch.Tensor,              # [B,T,H,S] softmaxed
        surprisal: Optional[torch.Tensor] = None,        # [B,T]
        longmass_base: Optional[torch.Tensor] = None,    # [B,H,T]
        longmass_delta: Optional[torch.Tensor] = None,   # [B,H,T]
    ):
        """
        Writes rows for this layer for the given batch.
        """

        assert gate1.dim() == 4 and gate2.dim() == 4, "gate1/gate2 must be [B,T,H,S]"
        assert input_ids.dim() == 2, "input_ids must be [B,T]"
        B, T, H, S = gate1.shape
        assert gate2.shape == (B, T, H, S)
        assert input_ids.shape == (B, T)

        if surprisal is not None:
            assert surprisal.shape == (B, T)
        if longmass_base is not None or longmass_delta is not None:
            assert longmass_base is not None and longmass_delta is not None
            assert longmass_base.shape == (B, H, T)
            assert longmass_delta.shape == (B, H, T)

        # rotate shard if needed
        if self._rows_written_in_shard >= self.cfg.max_rows_per_shard:
            self._shard_idx += 1
            self._open_new_shard()

        # ---- compute token position segment ----
        # pos_seg in [0..pos_segments-1] based on t/T
        segs = self.cfg.pos_segments
        # [T] on CPU
        t_idx = torch.arange(T, device=gate1.device)
        pos_seg = torch.clamp((t_idx.float() * segs / max(T, 1)).long(), 0, segs - 1).cpu()  # [T]

        # ---- flatten indexing ----
        # We'll write rows for each (b,t,h). Total rows = B*T*H.
        # Construct cpu tensors for necessary fields.

        token_id = input_ids.detach().to("cpu")  # [B,T]

        g1 = gate1.detach().float()
        g2 = gate2.detach().float()

        # top1 + top1p
        if self.cfg.record_top1:
            top1p1, top1_s1 = g1.max(dim=-1)  # [B,T,H]
            top1p2, top1_s2 = g2.max(dim=-1)
        else:
            top1p1 = top1_s1 = top1p2 = top1_s2 = None

        # entropy
        if self.cfg.record_entropy:
            # H = -sum p log p
            ent1 = -(g1 * torch.log(g1.clamp_min(self.cfg.eps))).sum(dim=-1)  # [B,T,H]
            ent2 = -(g2 * torch.log(g2.clamp_min(self.cfg.eps))).sum(dim=-1)
        else:
            ent1 = ent2 = None

        # mu (expected scale index)
        if self.cfg.record_mu:
            s_idx = self._get_scale_idx_cpu(S).to(g1.device)  # [S] on device
            mu1 = (g1 * s_idx).sum(dim=-1)  # [B,T,H]
            mu2 = (g2 * s_idx).sum(dim=-1)
        else:
            mu1 = mu2 = None

        # lowmass (last low_frac scales)
        if self.cfg.record_lowmass:
            low_start = int(round(S * (1.0 - self.cfg.low_frac)))
            low_start = max(0, min(S, low_start))
            lowmass1 = g1[..., low_start:].sum(dim=-1)  # [B,T,H]
            lowmass2 = g2[..., low_start:].sum(dim=-1)
        else:
            lowmass1 = lowmass2 = None

        # optional extras
        surp = surprisal.detach().to("cpu") if (self.cfg.record_surprisal and surprisal is not None) else None
        lm_base = longmass_base.detach().to("cpu") if (self.cfg.record_longmass and longmass_base is not None) else None
        lm_delta = longmass_delta.detach().to("cpu") if (self.cfg.record_longmass and longmass_delta is not None) else None

        # move computed tensors to cpu for writing
        def to_cpu(x):
            return x.detach().to("cpu") if x is not None else None

        top1p1 = to_cpu(top1p1); top1_s1 = to_cpu(top1_s1)
        top1p2 = to_cpu(top1p2); top1_s2 = to_cpu(top1_s2)
        ent1 = to_cpu(ent1); ent2 = to_cpu(ent2)
        mu1 = to_cpu(mu1); mu2 = to_cpu(mu2)
        lowmass1 = to_cpu(lowmass1); lowmass2 = to_cpu(lowmass2)

        # write rows
        # loop order chosen to keep python overhead moderate
        rows_written_now = 0
        for b in range(B):
            tok_row = token_id[b]  # [T]
            surp_row = surp[b] if surp is not None else None

            # [H,T] for longmass are stored [B,H,T]
            lm_base_b = lm_base[b] if lm_base is not None else None
            lm_delta_b = lm_delta[b] if lm_delta is not None else None

            for t in range(T):
                ps = int(pos_seg[t].item())
                tid = int(tok_row[t].item())
                s_val = float(surp_row[t].item()) if surp_row is not None else None

                # per-head fields (vector length H)
                if top1_s1 is not None:
                    s1_vec = top1_s1[b, t].tolist()
                    p1_vec = top1p1[b, t].tolist()
                    s2_vec = top1_s2[b, t].tolist()
                    p2_vec = top1p2[b, t].tolist()
                if ent1 is not None:
                    e1_vec = ent1[b, t].tolist()
                    e2_vec = ent2[b, t].tolist()
                if mu1 is not None:
                    m1_vec = mu1[b, t].tolist()
                    m2_vec = mu2[b, t].tolist()
                if lowmass1 is not None:
                    l1_vec = lowmass1[b, t].tolist()
                    l2_vec = lowmass2[b, t].tolist()
                if lm_base_b is not None:
                    # [H] at this t
                    lb_vec = lm_base_b[:, t].tolist()
                    ld_vec = lm_delta_b[:, t].tolist()

                for h in range(H):
                    row = [step, layer_idx, b, t, T, ps, h, tid]

                    if self.cfg.record_top1:
                        row += [int(s1_vec[h]), float(p1_vec[h]), int(s2_vec[h]), float(p2_vec[h])]
                    if self.cfg.record_entropy:
                        row += [float(e1_vec[h]), float(e2_vec[h])]
                    if self.cfg.record_mu:
                        row += [float(m1_vec[h]), float(m2_vec[h])]
                    if self.cfg.record_lowmass:
                        row += [float(l1_vec[h]), float(l2_vec[h])]

                    if self.cfg.record_surprisal:
                        row += [float(s_val) if s_val is not None else ""]
                    if self.cfg.record_longmass:
                        row += [float(lb_vec[h]) if lm_base_b is not None else "",
                                float(ld_vec[h]) if lm_delta_b is not None else ""]

                    self._writer.writerow(row)
                    rows_written_now += 1

        self._rows_written_total += rows_written_now
        self._rows_written_in_shard += rows_written_now

        if self._rows_written_total % self.cfg.flush_every < rows_written_now:
            self._f.flush()

        # rotate after writing if exceeded
        if self._rows_written_in_shard >= self.cfg.max_rows_per_shard:
            self._shard_idx += 1
            self._open_new_shard()

    def close(self, shutdown: bool = True):
        if self._f is not None:
            try:
                self._f.flush()
            except Exception:
                pass
            try:
                self._f.close()
            except Exception:
                pass
        self._f = None
        self._writer = None
        if shutdown:
            # write a tiny summary
            summary = {
                "rows_written_total": int(self._rows_written_total),
                "last_shard_idx": int(self._shard_idx),
            }
            with open(self.out / f"token_scale_summary{self._suffix()}.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)