# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.entmax import masked_entmax_bisect
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
    probs = torch.softmax(masked_logits, dim=dim)
    return torch.where(valid_mask, probs, torch.zeros_like(probs))


def causal_mask(batch_size: int, num_heads: int, q_len: int, k_len: int) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
    k_idx = torch.arange(k_len, device=device).view(1, 1, 1, k_len)
    return (k_idx <= q_idx).expand(batch_size, num_heads, q_len, k_len)


def test_masked_entmax_sum_to_one_and_non_negative():
    torch.manual_seed(42)
    batch_size, num_heads, q_len, k_len = 2, 3, 6, 6
    logits = torch.randn(batch_size, num_heads, q_len, k_len, device=device)
    mask = causal_mask(batch_size, num_heads, q_len, k_len)
    mask = mask & (torch.rand_like(logits) > 0.3)
    probs = masked_entmax_bisect(logits, alpha=1.2, valid_mask=mask, dim=-1)

    valid_rows = mask.any(dim=-1)
    sums = probs.sum(dim=-1)

    assert torch.all(probs >= 0)
    assert torch.all(probs[~mask] == 0)
    assert torch.allclose(sums[valid_rows], torch.ones_like(sums[valid_rows]), atol=1e-5, rtol=1e-5)


def test_masked_entmax_all_masked_row_returns_zeros():
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 5, device=device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[0, 2] = False

    probs = masked_entmax_bisect(logits, alpha=1.2, valid_mask=mask, dim=-1)

    assert torch.all(probs[0, 2] == 0)
    assert torch.isfinite(probs).all()


def test_masked_entmax_alpha_close_to_one_matches_masked_softmax():
    torch.manual_seed(42)
    batch_size, num_heads, q_len, k_len = 2, 2, 7, 7
    logits = torch.randn(batch_size, num_heads, q_len, k_len, device=device)
    mask = causal_mask(batch_size, num_heads, q_len, k_len)

    entmax = masked_entmax_bisect(logits, alpha=1.001, valid_mask=mask, dim=-1, n_iter=60)
    softmax = masked_softmax(logits.float(), mask, dim=-1).to(logits.dtype)

    assert torch.allclose(entmax.float(), softmax.float(), atol=1e-2, rtol=1e-2)


def test_masked_entmax_higher_alpha_is_sparser():
    logits = torch.tensor([[5.0, 4.5, 4.0, -1.0, -2.0, -3.0, -4.0, -5.0]], device=device)
    mask = torch.ones_like(logits, dtype=torch.bool)

    sparse = masked_entmax_bisect(logits, alpha=1.5, valid_mask=mask, dim=-1)
    dense = masked_entmax_bisect(logits, alpha=1.05, valid_mask=mask, dim=-1)
    softmax = torch.softmax(logits, dim=-1)

    sparse_zeros = (sparse == 0).sum().item()
    dense_zeros = (dense == 0).sum().item()
    softmax_zeros = (softmax == 0).sum().item()

    assert sparse_zeros > dense_zeros
    assert sparse_zeros > softmax_zeros


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_masked_entmax_dtype_round_trip(dtype: torch.dtype):
    torch.manual_seed(42)
    logits = torch.randn(2, 5, 8, device=device, dtype=dtype)
    mask = torch.rand(2, 5, 8, device=device) > 0.25

    probs = masked_entmax_bisect(logits, alpha=1.05, valid_mask=mask, dim=-1)

    assert probs.dtype == dtype
    assert torch.isfinite(probs.float()).all()


def test_masked_entmax_none_mask_matches_all_true_mask():
    torch.manual_seed(42)
    logits = torch.randn(3, 5, 4, device=device)
    mask = torch.ones_like(logits, dtype=torch.bool)

    none_mask = masked_entmax_bisect(logits, alpha=1.2, valid_mask=None, dim=1)
    true_mask = masked_entmax_bisect(logits, alpha=1.2, valid_mask=mask, dim=1)

    assert torch.allclose(none_mask, true_mask, atol=1e-6, rtol=1e-6)
