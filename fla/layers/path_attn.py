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
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.path_attn.parallel import parallel_path_attn

import math
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
            if config.wavelet_router:
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
                                print('router_non_linear_use', getattr(config, "router_non_linear_use", False))
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
    ) -> torch.Tensor:
        """
        Scale-wise routed:
        rel1 = sum_s (gate1 * q_s) P_s
        rel2 = sum_s (gate2 * qcorr_s) P_s
        rel  = rel1 - rel2   (or selection)
        Return: rel [B,H,T,T] (NO scale, NO mask)
        """
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
        shift_sep_use = bool(getattr(config, "shift_sep_use", False))
        if shift_sep_use:
            P_s = P.view(S, d_chunk, T, T)
            q_s = q0.view(B, T, H, S, d_chunk)
        else:
            P_s = P.view(S, d_chunk, T, T).mean(dim=1)
            q_s = q0.view(B, T, H, S, d_chunk).sum(dim=-1)

        # q_s: [B,T,H,S]  (组内求和；也可以改成 mean，看你定义)
        # 
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
            # gate1_c = gate1.unsqueeze(-1)
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
                q_corr = torch.einsum("b h t j, b j h d -> b t h d", M.to(compute_dtype), w0)
                if hier:
                    local_logits2 = self.local_router2(q_corr)  # [B,T,H,S]

                    qcorr_global = q_corr.reshape(B, T, H * D)
                    assert qcorr_global.shape[-1] == self.hidden_size, f"H*D={H*D} != hidden_size={self.hidden_size}"
                    global_logits2 = self.global_router2(qcorr_global).unsqueeze(2)  # [B,T,1,S]

                    mix_logits2 = local_logits2 + global_lambda * global_logits2
                    mix_logits2 = mix_logits2
                    gate2 = torch.softmax(mix_logits2, dim=-1)  # [B,T,H,S]
                qcorr_s = q_corr.view(B, T, H, S, d_chunk).sum(dim=-1)
                rel2 = torch.einsum("b t h s, s t n -> b h t n", gate2 * qcorr_s, P_s)
            else:
                q_corr = torch.einsum("b h t j, b j h d -> b t h d", M.to(compute_dtype), w0)
                qcorr_s = q_corr.view(B, T, H, S, d_chunk)
                gate2_c = gate2.unsqueeze(-1)
                rel2 = torch.einsum("b t h s c, s c t n -> b h t n", gate2_c * qcorr_s, P_s)

            if self.wavelet_coe is not None:
                rel2 = rel2 * self.wavelet_coe

        # combine
        if rel_selection == "rel1":
            rel = rel1
        elif rel_selection == "rel2":
            rel = -rel2
        elif rel_selection == "all":
            rel = rel1 - rel2
        else:
            raise ValueError(f"Unknown rel_selection={rel_selection}")

        # analyzer update: only when q_corr is available (or pass None and handle it in analyzer)
        if scale_wise_analyzer is not None and (E_base_raw is not None) and (q_corr is not None):
            # 这里传 gate/coe 你想记录什么都行；先沿用 rel2_coe
            scale_wise_analyzer.update(layer_idx, E_base_raw, q, q_corr, wavelet_dtt, coe=rel2_coe)

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

        # --- baseline attention ---
        E_base = E_base_raw * scale
        fill = causal_mask_fill_value(E_base.dtype)
        E_base = E_base.masked_fill(future, fill)
        P_base = torch.softmax(E_base, dim=-1)
        out_base = torch.einsum("b h i j, b j h d -> b i h d", P_base, v.to(compute_dtype))

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

        if wavelet_dtt is not None:
            if router1 is not None and router2 is not None:
                rel = self.wavelet_rel_from_M_scale_router(q, w, M_used, wavelet_dtt, compute_dtype=compute_dtype, d_chunk=d_chunk, layer_idx=layer_idx,
                                                    rel_selection=rel_selection, gate1=router1, gate2=router2,config=config)
            else:
                # rel = wavelet_rel_from_M(q, w, M_used, wavelet_dtt, compute_dtype=compute_dtype,layer_idx=layer_idx, rel_selection=rel_selection, rel1_coe=rel1_coe, rel2_coe=rel2_coe, scale_wise_analyzer=scale_wise_analyzer,
                #                          E_base_raw=E_base_raw if scale_wise_analyzer is not None else None)
                rel = wavelet_rel_from_M(q, w, M_used, wavelet_dtt, compute_dtype=compute_dtype,layer_idx=layer_idx, rel_selection=rel_selection,
                                        E_base_raw=E_base_raw if analyzer is not None else None)

            # optional ablation on rel
            if ablate is not None and layer_idx in ablate:
                for h in ablate[layer_idx]:
                    rel[:, h].zero_()
            E_wav_raw = E_base_raw + rel
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
        P_wav = torch.softmax(E_wav, dim=-1)
        out_wav = torch.einsum("b h i j, b j h d -> b i h d", P_wav, v.to(compute_dtype))

        if analyzer is not None and rel is not None:
            w0 = w.to(compute_dtype)
            deltaQ = torch.einsum("b h i j, b j h d -> b i h d", M_used, w0)
            analyzer['layer_attention_analyzer'].update(layer_idx, E_base_raw, rel, P_base, P_wav, q, deltaQ)

        pwav_logger = analyzer.get("pwav_mean_logger") if isinstance(analyzer, dict) else None
        if pwav_logger is not None:
            pwav_logger.update(layer_idx, out_wav)
        return out_base, out_wav
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
        router1, router2 = None, None
        record_router1, record_router2 = None, None
        if self.config.wavelet_router:
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
                    tau = 1.0 if not hasattr(self.config, "tau") else float(self.config.tau)
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
                        # ---- (stat monitor) log in BOTH train & eval ----
                        # =========================
                        def _should_log() -> bool:
                            try:
                                return (int(global_step) % log_every == 0)
                            except Exception:
                                return True

                        def _fmt_vec(x: torch.Tensor, nd: int = 4) -> str:
                            x = x.detach().float().cpu().tolist()
                            return "[" + ",".join([f"{v:.{nd}f}" for v in x]) + "]"

                        do_log = _should_log()
                        want_log = (self.training and log_router_stats_train) or ((not self.training) and log_router_stats_eval)

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

                    # def _add_gaussian_jitter(logits: torch.Tensor, std: float) -> torch.Tensor:
                    #     if std <= 0:
                    #         return logits
                    #     if (not self.training) and (not jitter_apply_in_eval):
                    #         return logits

                    #     # ---- config ----
                    #     noise_adapt_style = getattr(self.config, "noise_adapt_style", 'logit_std')
                    #     # optional: "const" | "logit_std" | "fliprate_head"
                    #     # backward compatible default:

                    #     sigma_max = getattr(self.config, "router_jitter_sigma_max", None)
                    #     sigma_max = float("inf") if sigma_max is None else float(sigma_max)

                    #     # (optional but recommended for stability)
                    #     sigma_min = getattr(self.config, "router_jitter_sigma_min", None)
                    #     sigma_min = 0.0 if sigma_min is None else float(sigma_min)

                    #     # ---- compute sigma_eff ----
                    #     if noise_adapt_style == "fliprate_head":
                    #         # Interpret `std` as target flip rate rho0 by default.
                    #         # If you want rho0 from config instead, uncomment next line:
                    #         # rho0 = float(getattr(self.config, "router_jitter_target_flip", std))
                    #         rho0 = float(std)

                    #         # sanity range; avoid icdf inf/nan
                    #         rho0 = min(max(rho0, 1e-6), 0.499999)

                    #         # margin m(b,t,h) = top1 - top2 over S
                    #         # detach to avoid noisy gradients through calibration
                    #         logits_det = logits.detach()
                    #         B, T, H, S = logits_det.shape
                    #         if S < 2:
                    #             # routing with <2 choices doesn't make sense for margin-based flip calibration
                    #             return logits

                    #         top2 = torch.topk(logits_det, k=2, dim=-1).values  # [B,T,H,2]
                    #         margin = (top2[..., 0] - top2[..., 1]).clamp_min(1e-6)  # [B,T,H]

                    #         # head-wise aggregation over (B,T): use median (robust)
                    #         # shape -> [BT, H]
                    #         margin_bt = margin.reshape(-1, H)
                    #         margin_head = margin_bt.median(dim=0).values  # [H]

                    #         # sigma_head = margin_head / (sqrt(2)*|Phi^{-1}(rho0)|)
                    #         normal = torch.distributions.Normal(
                    #             loc=logits_det.new_tensor(0.0),
                    #             scale=logits_det.new_tensor(1.0),
                    #         )
                    #         z = normal.icdf(logits_det.new_tensor(rho0)).abs().clamp_min(1e-6)
                    #         denom = (math.sqrt(2.0) * z)

                    #         sigma_head = (margin_head / denom)  # [H]
                    #         sigma_head = sigma_head.clamp(min=sigma_min, max=sigma_max)

                    #         # broadcast to [B,T,H,1]
                    #         sigma_eff = sigma_head.view(1, 1, H, 1)

                    #     elif noise_adapt_style == "logit_std":
                    #         scale = logits.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,T,H,1]
                    #         sigma_eff = (float(std) * scale).clamp(min=sigma_min, max=sigma_max)

                    #     elif noise_adapt_style == "const":
                    #         sigma_eff = logits.new_full(logits.shape[:-1] + (1,), float(std))
                    #         sigma_eff = sigma_eff.clamp(min=sigma_min, max=sigma_max)

                    #     else:
                    #         raise ValueError(f"Unknown noise_adapt_style: {noise_adapt_style}")
                    #     noise = torch.randn_like(logits) * sigma_eff  # [B,T,H,S]

                    #     if log_noise_stats:
                    #         try:
                    #             noise_det = noise.detach()
                    #             noise_mean = noise_det.mean().item()
                    #             noise_std_val = noise_det.std().item()
                    #             noise_min = noise_det.min().item()
                    #             noise_max = noise_det.max().item()
                    #             msg = (
                    #                 f"[router jitter noise] layer={self.layer_idx} "
                    #                 f"step={global_step}/{max_steps} mean={noise_mean:.6f} "
                    #                 f"std={noise_std_val:.6f} min={noise_min:.6f} max={noise_max:.6f} "
                    #                 f"style={noise_adapt_style}"
                    #             )
                    #             logger_obj = getattr(self, "logger", None)
                    #             if logger_obj is not None:
                    #                 try:
                    #                     logger_obj.info(msg)
                    #                 except Exception:
                    #                     print(msg)
                    #             else:
                    #                 print(msg)
                    #         except Exception:
                    #             pass

                    #     return logits + noise

                    # def _add_gaussian_jitter(logits: torch.Tensor, std: float) -> torch.Tensor:
                    #     if std <= 0:
                    #         return logits
                    #     if (not self.training) and (not jitter_apply_in_eval):
                    #         return logits

                    #     sigma_max = getattr(self.config, "router_jitter_sigma_max", None)
                    #     if sigma_max is None:
                    #         sigma_max = float("inf")
                    #     else:
                    #         sigma_max = float(sigma_max)

                    #     if jitter_scale_by_logit_std:
                    #         scale = logits.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)   # [B,T,H,1]
                    #         sigma_eff = (std * scale).clamp_max(sigma_max)                      # [B,T,H,1]
                    #     else:
                    #         # constant sigma for all (B,T,H); broadcast over S
                    #         sigma_eff = logits.new_full(logits.shape[:-1] + (1,), float(std)).clamp_max(sigma_max)
                    #     noise = torch.randn_like(logits) * sigma_eff                             # [B,T,H,S]
                    #     if log_noise_stats:
                    #         try:
                    #             noise_det = noise.detach()
                    #             noise_mean = noise_det.mean().item()
                    #             noise_std_val = noise_det.std().item()
                    #             noise_min = noise_det.min().item()
                    #             noise_max = noise_det.max().item()
                    #             msg = (
                    #                 f"[router jitter noise] layer={self.layer_idx} "
                    #                 f"step={global_step}/{max_steps} mean={noise_mean:.6f} "
                    #                 f"std={noise_std_val:.6f} min={noise_min:.6f} max={noise_max:.6f}"
                    #             )
                    #             logger_obj = getattr(self, "logger", None)
                    #             if logger_obj is not None:
                    #                 try:
                    #                     logger_obj.info(msg)
                    #                 except Exception:
                    #                     print(msg)
                    #             else:
                    #                 print(msg)
                    #         except Exception:
                    #             pass

                    #     return logits + noise

                    # add jitter BEFORE softmax
                    router1_logits = _add_gaussian_jitter(router1_logits, jitter_std, router_name="router1")
                    router1 = torch.softmax(router1_logits / tau, dim=-1)  # [B,T,H,S]

                    if self.router2 is None:
                        router2 = None
                    else:
                        router2_logits = self.router2(hidden_states)       # [B,T,H*S]
                        router2_logits = router2_logits.view(B, T, H, S)   # [B,T,H,S]
                        router2_logits = _add_gaussian_jitter(router2_logits, jitter_std, router_name="router2")
                        router2 = torch.softmax(router2_logits / tau, dim=-1)  # [B,T,H,S]
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
                
                out_base, o = self.path_attention_with_wavelet_QH(
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
                    router1=router1 if self.config.wavelet_router else None,
                    router2=router2 if self.config.wavelet_router else None,
                    config=self.config,
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