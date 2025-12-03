# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.distributed as dist

from fla.layers.utils import pad_input, unpad_input
# from fla.layers.freq_analysis_utils import spectrum_stats_from_logits, spectrum_over_T_multi
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

import pdb
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


def spectral_distill_over_L(
    student: torch.Tensor,   # [B, L, H, D]  (path_attn_scores 映射/reshape到该形状)
    teacher: torch.Tensor,   # [B, L, H, D]  (rotary 或 wavelet 的 logits 映射/reshape)
    *,
    tau: float = 1.0,                      # 暂时不用，保留接口
    w_band: torch.Tensor | None = None,    # [K] 可选频带权重
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

    # 频带加权（可选）
    if w_band is not None:
        # w_band: [K] -> [1,1,K,1,1]
        w = w_band.to(A_s).view(1, 1, -1, 1, 1)
    else:
        w = 1.0

    # 2) 在 K 维上做归一化，得到“频率分布” p_t, p_s
    # 先保证非负（幅值本身就是非负，这里只是稳一手）
    # A_t_clamp = A_t.clamp_min(0.0)
    # A_s_clamp = A_s.clamp_min(0.0)

    # # sum over K: [B,Q,1,H,D]
    # sum_t = A_t_clamp.sum(dim=2, keepdim=True)
    # sum_s = A_s_clamp.sum(dim=2, keepdim=True)

    # p_t = A_t_clamp / (sum_t + eps)  # [B,Q,K,H,D]
    # p_s = A_s_clamp / (sum_s + eps)  # [B,Q,K,H,D]

    # 3a) KL(p_t || p_s)
    # if lambda_kl != 0.0:
    #     kl = p_t * ((p_t + eps).log() - (p_s + eps).log())
    #     loss_kl = (kl * w).mean()
    # else:
    #     loss_kl = A_s.new_tensor(0.0)

    # # 3b) log-prob MSE（形状 MSE）
    # if lambda_mse != 0.0:
    #     log_p_t = (p_t + eps).log()
    #     log_p_s = (p_s + eps).log()
    #     loss_mse = (((log_p_s - log_p_t) ** 2) * w).mean()
    # else:
    #     loss_mse = A_s.new_tensor(0.0)
    eps = 1e-8

    # A_t_clamp = A_t.clamp_min(eps)
    # A_s_clamp = A_s.clamp_min(eps)

    # sum_t = A_t_clamp.sum(dim=2, keepdim=True)  # K 维是 dim=2
    # sum_s = A_s_clamp.sum(dim=2, keepdim=True)

    # p_t = A_t_clamp / (sum_t + eps)
    # p_s = A_s_clamp / (sum_s + eps)

    # log_p_t = (p_t + eps).log()
    # log_p_s = (p_s + eps).log()

    loss_spec_mse = ((A_t_log - A_s_log) ** 2 * w).mean()
    return loss_spec_mse
    # loss = lambda_kl * loss_kl + lambda_mse * loss_mse

    # return loss

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
    # if show_progress:
    #     t_iter = tqdm(t_iter, desc="path Householder steps")

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

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        # w 分支输出扩到 H*R*d
        out_w = self.kv_dim * self.r
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w, bias=False)
            )
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
        if self.config.distill_teacher == 'rotary':
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
            wave = None
            if getattr(self, "use_wavelet_beta", False):
                B, T = hidden_states.shape[:2]
                # pos ∈ [-T+1, ..., 0]，末位为中心
                pos_end = torch.arange(0, T, device=hidden_states.device).unsqueeze(0).to(hidden_states.dtype)  # [1,T,1,1]

                # 指数项可学：scale = 2**e，e 初始为负序列，窗更宽/衰减更慢
                e = self.ricker_scale_exp.to(hidden_states.dtype)                          # [1,1,H,r]
                scale = torch.exp2(e)                                                      # [1,1,H,r]
                shift = self.ricker_shift.to(hidden_states.dtype)                          # [1,1,H,r]
                t_affine = scale * (pos_end - shift)                                       # [B,T,H,r]

                # Ricker（无 σ 版本）
                psi = (1.0 - t_affine**2) * torch.exp(-0.5 * t_affine**2)     
                wave = (psi - psi.min(dim=1, keepdim=True)[0]) / (psi.max(dim=1, keepdim=True)[0] - psi.min(dim=1, keepdim=True)[0] + 1e-6)             # [B,T,H,r]
            beta = torch.sigmoid(beta_logits) * 2.0
            # per-head logging（参数 + 运行时）
            
            # if torch.cuda.current_device() == 0:
            #     self.steps += 1
            #     if (self.logging_steps > 0) and (self.steps % self.logging_steps == 0):
            #         try:
            #             ratio_head = self.path_attention_ratio if self.wavelet_baseline_use else torch.tensor(1.0, device=hidden_states.device)
            #             vals = ratio_head.detach().cpu().tolist() if ratio_head.dim() > 0 else float(ratio_head)
            #             print(f"layer{self.layer_idx}: path attention ratio: {vals}")
            #         except Exception as e:
            #             ratio_head    = self.path_attention_ratio if self.wavelet_baseline_use else torch.tensor(1.0)

            #             vals = ratio_head.detach().cpu().tolist()
            #             print(f"layer{self.layer_idx}: path attention ratio: {vals}")

            # g（若开启）扩到 R
            if g is not None:
                g = rearrange(g, 'b t hq -> b t hq 1').repeat(1, 1, self.r, 1).view(g.shape[0], g.shape[1], -1)

            # 核心 op

            o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
            if (self.layer_idx < self.config.distill_in_which_layers) and self.training:
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
                        temp_teacher_scores = rot_q[batch_idx, j_idx, ...] * rot_k[batch_idx, i_idx, ...]
                        spectral_teacher_scores = rot_q[:, -1:, ...] * rot_k
                    elif self.config.distill_teacher == 'wavelet':
                        spectral_teacher_scores = compute_wavelet_scores_batched(q[:, -1, ...], k, wavelet_decay_table[:, -1, :])
                        temp_teacher_scores = compute_pair_wavelet_scores_batched(q[batch_idx, j_idx, ...], k[batch_idx, i_idx, ...], wavelet_decay_table[:, j_idx, i_idx])                      
                    else:
                        raise ValueError(f"Unknown distill_teacher: {self.config.distill_teacher}")
                temp_path_attn_scores = compute_path_score_multi(
                                                                    q, k, w, beta,
                                                                    i_idx=i_idx,
                                                                    j_idx=j_idx,
                                                                    batch_idx=batch_idx,
                                                                )                  
                path_attn_scores = path_attn_last_query_elementwise(q[:, -1:, ...], k, w, beta)
                spectral_loss = spectral_distill_over_L(path_attn_scores.unsqueeze(1), spectral_teacher_scores.unsqueeze(1), lambda_kl=0.0, lambda_mse=1.0)
 
                temp_loss = F.mse_loss(temp_path_attn_scores, temp_teacher_scores)
                dis_loss = self.config.temp_loss_coe * temp_loss + self.config.spectral_loss_coe * spectral_loss
            else:
                dis_loss = torch.tensor(0.0, device=q.device, dtype=q.dtype)
            o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
            o = self.o_proj(o)
            return o, None, past_key_values, dis_loss

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