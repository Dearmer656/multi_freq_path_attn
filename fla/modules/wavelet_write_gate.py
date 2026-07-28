from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn


def _ricker_wavelet(u):
    return (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))


def _rms_norm_last_dim(x, eps=1e-6):
    xf = x.to(dtype=torch.float32)
    denom = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True).clamp_min(0.0) + float(eps))
    return xf / denom


def _maybe_clamp_p99(x, *, enable, quantile, scale, min_val, max_samples=100000):
    if not enable:
        return x.to(dtype=torch.float32)
    x32 = x.to(dtype=torch.float32)
    abs_flat = x32.detach().abs().reshape(-1)
    if abs_flat.numel() == 0:
        return x32
    q = min(max(float(quantile), 0.5), 0.9999)
    if abs_flat.numel() > max_samples:
        idx = torch.linspace(0, abs_flat.numel() - 1, steps=max_samples, device=abs_flat.device).long()
        abs_eval = abs_flat.index_select(0, idx)
    else:
        abs_eval = abs_flat
    clamp_ref = torch.quantile(abs_eval, q)
    clamp_v = torch.clamp(clamp_ref * float(scale), min=float(min_val))
    return x32.clamp(min=-clamp_v, max=clamp_v)


def _make_scales(scale_max_exp: Union[float, list[float]], k_scales: int):
    if k_scales == 1:
        exp = float(scale_max_exp) if not isinstance(scale_max_exp, list) else float(scale_max_exp[0])
        return [2.0 ** (exp / 2.0)]
    if isinstance(scale_max_exp, list):
        if len(scale_max_exp) != k_scales:
            raise ValueError("scale_max_exp must have length k_scales when provided as a list")
        return [2.0 ** (float(e) / 2.0) for e in scale_max_exp]
    return [2.0 ** (float(scale_max_exp) / 2.0) for _ in range(k_scales)]


class WaveletWriteGate(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        k_scales: int,
        scale_max_exp,
        sigmoid_mode: str,
        tau: float,
        rms_eps: float,
        clamp1_enable: bool,
        clamp1_quantile: float,
        clamp1_scale: float,
        clamp1_min: float,
        g_bias_max: float,
        layer_gain_init: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.k_scales = k_scales
        self.sigmoid_mode = sigmoid_mode
        self.tau = float(tau)
        self.rms_eps = float(rms_eps)
        self.clamp1_enable = clamp1_enable
        self.clamp1_quantile = float(clamp1_quantile)
        self.clamp1_scale = float(clamp1_scale)
        self.clamp1_min = float(clamp1_min)
        self.g_bias_max = float(g_bias_max)

        scales = torch.tensor(_make_scales(scale_max_exp, k_scales), dtype=torch.float32)
        self.register_buffer("scales", scales, persistent=False)

        self.layer_norm = nn.LayerNorm(head_dim)
        self.router = nn.Linear(head_dim, k_scales + 1)
        self.layer_gain_raw = nn.Parameter(torch.tensor(float(layer_gain_init)))

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        h = hidden_states.detach().view(batch_size, seq_len, self.num_heads, self.head_dim).mean(dim=2)
        feat = self.layer_norm(h)
        router_logits = self.router(feat)
        router_logits = _rms_norm_last_dim(router_logits, eps=self.rms_eps)
        tau = self.tau

        if self.sigmoid_mode == "signed":
            pi_scale = 2.0 * torch.sigmoid(router_logits[..., 1:] / tau) - 1.0
        elif self.sigmoid_mode == "with_null":
            g = torch.sigmoid(router_logits[..., 1:] / tau)
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(self.rms_eps)
            w = g / sum_g
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau)
            pi_scale = g0_gate * w
        elif self.sigmoid_mode == "with_null_independent_scales":
            # Factorized routing (matches fla/layers/path_attn.py): null vs
            # non-null compete through g0_gate, but scales do NOT compete
            # against each other inside non-null (no renormalization by sum_g),
            # so multiple scales can be simultaneously active.
            g = torch.sigmoid(router_logits[..., 1:] / tau)
            g0_gate = torch.sigmoid(router_logits[..., 0:1] / tau)
            pi_scale = g0_gate * g
        else:
            raise ValueError(f"unsupported mode {self.sigmoid_mode}")

        pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
        basis = []
        for i in range(self.k_scales):
            u_i = pos / self.scales[i]
            basis_i = _ricker_wavelet(u_i)
            basis_i = _rms_norm_last_dim(basis_i, eps=self.rms_eps)
            basis_i = _maybe_clamp_p99(
                basis_i,
                enable=self.clamp1_enable,
                quantile=self.clamp1_quantile,
                scale=self.clamp1_scale,
                min_val=self.clamp1_min,
            )
            basis.append(basis_i)
        basis_stack = torch.stack(basis, dim=-1)
        wavelet_bias = torch.einsum("btk,tk->bt", pi_scale, basis_stack)
        g_layer = torch.nn.functional.softplus(self.layer_gain_raw)
        wavelet_bias = wavelet_bias * g_layer
        wavelet_bias = wavelet_bias.clamp(min=-self.g_bias_max, max=self.g_bias_max)
        # Return the raw bias only; callers apply B_t' = B_t * (1 + bias)
        # explicitly, so the "+1" stays visible at the point B is gated.
        return wavelet_bias
