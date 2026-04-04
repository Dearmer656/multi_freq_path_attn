"""
Test: parallel_path_attn with return_logits=True

Expected logit_out [B, HQ, T, T]:
  - Upper triangle (j >= i): -inf  (future, causal mask)
  - Within-same-BS-block positions (j in [block_start, i)): -inf  (handled by intra_chunk_preprocess, NOT stored)
  - Cross-block causal (j < block_start(i)): finite

With w=0 (no path state updates), cross-BT-chunk logits should equal standard QK^T * scale.
"""

import torch
import math
import sys
sys.path.insert(0, '/project/nlp-work5/hongyu-s/flash-linear-attention')

from fla.ops.path_attn.parallel import parallel_path_attn

torch.manual_seed(0)
device = 'cuda'
dtype = torch.bfloat16

B, T, H, K, V_dim = 2, 128, 4, 64, 64  # B=2 to cover multi-batch bug
HQ = H  # no GQA for simplicity
scale = K ** -0.5

# BT=64, BS=32 (determined by check_shared_mem inside the fn)
# On non-hopper: BS=32, on non-ampere: BT=64
BT = 64
BS = 32

print(f"B={B}, T={T}, BT={BT}, BS={BS}, K={K}")

q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
k = torch.randn(B, T, H,  K, device=device, dtype=dtype)
v = torch.randn(B, T, H,  V_dim, device=device, dtype=dtype)
# w, beta must be float32
w    = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
beta = torch.ones( B, T, H,    device=device, dtype=torch.float32)

# ── Test 1: shape ──────────────────────────────────────────────────────────────
logits, k_cache = parallel_path_attn(q, k, v, w, beta, scale=scale, return_logits=True)
assert k_cache is None, f"Expected k_cache=None when return_logits=True, got {k_cache}"
assert logits.shape == (B, HQ, T, T), f"Expected {(B, HQ, T, T)}, got {logits.shape}"
print(f"[PASS] shape: {logits.shape}")

# ── Test 2: strict upper triangle is -inf ──────────────────────────────────────
# logit[b, h, i, j] should be -inf for j >= i
idx_q = torch.arange(T, device=device)
idx_k = torch.arange(T, device=device)
upper = idx_k[None, :] >= idx_q[:, None]  # [T, T], True where j >= i

logits_h0 = logits[0, 0]  # [T, T]
upper_vals = logits_h0[upper]
assert (upper_vals == float('-inf')).all(), \
    f"Upper triangle contains non-inf values: max={upper_vals.max()}"
print(f"[PASS] upper triangle is all -inf ({upper.sum().item()} positions)")

# ── Test 3: within-same-BS-block positions are -inf ────────────────────────────
# For query i in BS block b (b*BS <= i < (b+1)*BS),
# keys j in [b*BS, i) are handled by intra_chunk_preprocess, NOT in logit_out → -inf
within_block_count = 0
for i in range(T):
    block_start = (i // BS) * BS
    for j in range(block_start, i):  # same BS block, before i
        val = logits_h0[i, j].item()
        if val != float('-inf'):
            print(f"  [FAIL] logit[{i},{j}] = {val:.4f}, expected -inf (within BS block)")
            within_block_count += 1

if within_block_count == 0:
    same_block_pairs = sum((i - (i // BS) * BS) for i in range(T))
    print(f"[PASS] all {same_block_pairs} within-BS-block causal positions are -inf (not captured by logit_out)")
else:
    print(f"[FAIL] {within_block_count} within-BS-block positions are NOT -inf")

# ── Test 4: cross-block causal positions are finite ─────────────────────────────
finite_count = 0
inf_unexpected = 0
for i in range(T):
    block_start = (i // BS) * BS
    for j in range(0, block_start):  # cross-block causal
        val = logits_h0[i, j].item()
        if math.isfinite(val):
            finite_count += 1
        else:
            inf_unexpected += 1

print(f"[{'PASS' if inf_unexpected == 0 else 'FAIL'}] cross-block causal: {finite_count} finite, {inf_unexpected} unexpected non-finite")

# ── Test 4b: B>1 — second batch must also have finite cross-block values ───────
finite_b1 = 0
for i in range(T):
    block_start = (i // BS) * BS
    for j in range(0, block_start):
        if math.isfinite(logits[1, 0, i, j].item()):
            finite_b1 += 1
expected_cross = sum((i // BS) * BS for i in range(T))
print(f"[{'PASS' if finite_b1 == expected_cross else 'FAIL'}] batch-1 cross-block: {finite_b1} finite (expected {expected_cross})")

# ── Test 5: w=0 → logits equal standard QK^T * scale * log2(e) ────────────────
# IMPORTANT: The kernel stores logits in base-2 scale:
#   sm_scale = scale * log2(e)  so that  exp2(b_s - max) = exp(score*scale - max)
# So: returned logit = (q @ k.T) * scale * log2(e)
# To recover standard attention weights: softmax(logits * ln(2), dim=-1)
#   because softmax(x) = exp2(x) / sum(exp2(x)) = exp(x*ln2) / sum(exp(x*ln2))
print("\n--- w=0 consistency check (logits should be QK^T * scale * log2(e)) ---")
LOG2E = math.log2(math.e)  # 1.44269504
w_zero = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
logits_w0, _ = parallel_path_attn(q, k, v, w_zero, beta, scale=scale, return_logits=True)

q_fp32 = q.float().permute(0, 2, 1, 3)  # [B, HQ, T, K]
k_fp32 = k.float().permute(0, 2, 1, 3)  # [B, H,  T, K]
std_scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * scale  # [B, HQ, T, T]
causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
expected_logits = std_scores.masked_fill(~causal_mask, float('-inf')) * LOG2E

# Compare only cross-BT positions
cross_bt_count = 0
max_diff = 0.0
for i in range(T):
    bt_start = (i // BT) * BT
    for j in range(0, bt_start):
        v0 = logits_w0[0, 0, i, j].item()
        v1 = expected_logits[0, 0, i, j].item()
        if math.isfinite(v0) and math.isfinite(v1):
            diff = abs(v0 - v1)
            max_diff = max(max_diff, diff)
            cross_bt_count += 1

print(f"Cross-BT positions compared: {cross_bt_count}")
print(f"Max |logit_w0 - QKT*scale*log2e| = {max_diff:.6f}")
if max_diff < 0.05:
    print("[PASS] w=0 logits match QK^T * scale * log2(e) at cross-BT positions")
else:
    print("[FAIL] larger than expected diff")
    shown = 0
    for i in range(BT, min(2*BT, T)):
        for j in range(0, BT, 8):
            if j < i:
                v0 = logits_w0[0, 0, i, j].item()
                v1 = expected_logits[0, 0, i, j].item()
                print(f"    ({i:3d},{j:3d}): logit={v0:.4f} expected={v1:.4f}")
                shown += 1
                if shown >= 8:
                    break
        if shown >= 8:
            break

print("\nNOTE: to use logits as attention weights:")
print("  attn_weights = torch.softmax(logits * math.log(2), dim=-1)  # convert base-2 to base-e")

# ── Test 6: visualise a small slice ───────────────────────────────────────────
print("\n--- Logit heatmap slice [32:48, 0:48] head=0 (w=default) ---")
slice_q = slice(32, 48)
slice_k = slice(0, 48)
sl = logits[0, 0, slice_q, slice_k].float()
for row_i, row in enumerate(sl):
    vals = []
    for v_val in row:
        if v_val.item() == float('-inf'):
            vals.append("  -inf")
        else:
            vals.append(f"{v_val.item():6.2f}")
    print(f"  q={32+row_i:3d}: " + " ".join(vals))

print("\nAll tests done.")
