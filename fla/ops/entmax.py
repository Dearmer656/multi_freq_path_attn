# -*- coding: utf-8 -*-

"""Masked alpha-entmax via bisection for PAT-195 test-time attention normalization.

This implementation is causal-mask friendly and runs the fragile exponentiation in fp32.
"""

import torch


def _bisect_tau(
    y: torch.Tensor,
    valid_mask: torch.Tensor | None,
    dim: int,
    n_iter: int,
    *,
    power: float,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(y, dtype=torch.bool)
    else:
        valid_mask = torch.broadcast_to(valid_mask.to(device=y.device, dtype=torch.bool), y.shape)

    neg_inf = torch.finfo(y.dtype).min
    y_masked = torch.where(valid_mask, y, torch.full_like(y, neg_inf))
    valid_any = valid_mask.any(dim=dim, keepdim=True)
    row_max = torch.where(valid_any, y_masked.amax(dim=dim, keepdim=True), torch.zeros_like(y_masked.sum(dim=dim, keepdim=True)))

    # For entmax, tau lives in [max(y) - 1, max(y)] after scaling y = (alpha - 1) * logits.
    tau_lo = row_max - 1
    tau_hi = row_max

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) * 0.5
        p_mid = torch.relu(y - tau_mid).pow(power)
        p_mid = torch.where(valid_mask, p_mid, torch.zeros_like(p_mid))
        mass = p_mid.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(mass > 1, tau_mid, tau_lo)
        tau_hi = torch.where(mass > 1, tau_hi, tau_mid)

    return (tau_lo + tau_hi) * 0.5


def masked_entmax_bisect(
    logits: torch.Tensor,
    *,
    alpha: float,
    valid_mask: torch.Tensor | None,
    dim: int = -1,
    n_iter: int = 50,
    eps: float = 1e-6,
) -> torch.Tensor:
    if alpha <= 1:
        raise ValueError(f"alpha must be > 1, got {alpha}")

    dim = dim if dim >= 0 else logits.dim() + dim
    logits_fp32 = logits.float()
    # alpha close to 1 yields a large exponent, so keep the whole solve in fp32.
    power = 1.0 / (alpha - 1.0)
    y = (alpha - 1.0) * logits_fp32

    if valid_mask is None:
        mask = torch.ones_like(logits_fp32, dtype=torch.bool)
    else:
        mask = torch.broadcast_to(valid_mask.to(device=logits.device, dtype=torch.bool), logits.shape)

    tau = _bisect_tau(y, mask, dim, n_iter, power=power)
    probs = torch.relu(y - tau).pow(power)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs_sum = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(probs_sum > eps, probs / probs_sum.clamp_min(eps), torch.zeros_like(probs))
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    return probs.to(logits.dtype)
