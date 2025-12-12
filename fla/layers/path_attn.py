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


import pdb
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


import os
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def plot_out_head_dim_groups_or_grouped(
    out: torch.Tensor,
    save_dir: str = "plots_out_traces",
    name: str = "",
    group_size: int = 8,
    separate_plots: bool = False,
    distill_teacher: str = "wavelet",  # "wavelet" or "rotary"
):
    """
    支持两种输入：

    1) out.shape == [B, T, H, D]
       - distill_teacher == "wavelet":
           对 D 维按 group_size 分组 (默认 8 维一组)，在组内平均，
           得到 [B, T, H, S]，再在 B 上平均 -> [T, H, S]，
           最后画 head×S 条曲线（K=T 为横轴）。
       - distill_teacher == "rotary":
           不分组，直接每 group_size 维取一个 dim (0, group_size, 2*group_size, ...)，
           得到 [B, T, H, S]，在 B 上平均 -> [T, H, S]，画这些具体 dim 的曲线。

    2) out.shape == [H, S, K]
       - 认为已经分好组，无需再 group/平均，
         直接把 K 作为横轴画 head×S 条曲线。

    Args:
        out: Tensor, shape [B, T, H, D] 或 [H, S, K]
        name: 文件名前缀，用于区分不同实验
        save_dir: 保存图片的目录
        group_size: 
            wavelet: 每组的大小（默认 8 维一组）
            rotary : 作为 stride，每 group_size 维取一个 dim（默认每 8 维取一个）
        separate_plots:
            True: 每个 (head, group/dim) 一张图 -> H * S 张
            False: 每个 head 一张图，里面画多个 group/dim 的曲线 -> H 张
        distill_teacher:
            "wavelet": 使用分组平均显示 scale
            "rotary" : 每 group_size 维取一个具体 dim 显示
    """
    os.makedirs(save_dir, exist_ok=True)

    if out.dim() == 4:
        # --------- 情况 1: [B, T, H, D] ---------
        B, T, H, D = out.shape

        if distill_teacher == "wavelet":
            # [B, T, H, D] -> [B, T, H, S, group_size]
            assert D % group_size == 0, f"D={D} 必须能被 group_size={group_size} 整除"

            S = D // group_size
            out_grouped = out.view(B, T, H, S, group_size)

            # 在 group_size 上平均 -> [B, T, H, S]
            out_group_mean = out_grouped.mean(dim=-1)

            # 在 batch 上平均 -> [T, H, S]
            data = out_group_mean.mean(dim=0).cpu().numpy()
            K_len = T
            x_axis = range(K_len)

            group_labels = [
                f"group{g} (dims {g*group_size}-{(g+1)*group_size-1})"
                for g in range(S)
            ]

        elif distill_teacher in ["rotary", "shrink", "shrink_w_shuffle"]:
            # 每 group_size 维取一个具体 dim：0, group_size, 2*group_size, ...
            assert group_size > 0, "group_size 必须为正整数"
            device = out.device
            selected_indices = torch.arange(0, D, group_size, device=device)  # [S]
            S = selected_indices.numel()

            # 选出这些维度: [B, T, H, S]
            out_selected = out[..., selected_indices]

            # 在 batch 上平均 -> [T, H, S]
            data = out_selected.mean(dim=0).cpu().numpy()
            K_len = T
            x_axis = range(K_len)

            idx_list = selected_indices.tolist()
            group_labels = [
                f"dim{d_idx}"
                for d_idx in idx_list
            ]
        else:
            raise ValueError(f"未知的 distill_teacher='{distill_teacher}'，应为 'wavelet' 或 'rotary'")

    elif out.dim() == 3:
        # --------- 情况 2: [H, S, K]，已经分好组 ---------
        H, S, K_len = out.shape
        data = out.cpu().numpy()          # [H, S, K]
        x_axis = range(K_len)

        group_labels = [
            f"group{g}"
            for g in range(S)
        ]
    else:
        raise ValueError(
            f"out 维度必须是 3 或 4，当前形状 {out.shape} (dim={out.dim()})"
        )

    # --------- 统一画图逻辑 ---------
    # 对于 4 维输入，此时 data.shape == [T, H, S]
    # 我们想要 [K, H, S] 的风格，K 是横轴
    if out.dim() == 4:
        # data: [T, H, S] -> [K, H, S]，这里 K=T
        data = data  # [K, H, S]
    else:
        # 3 维时 data: [H, S, K] -> [K, H, S]，方便统一处理
        data = data.transpose(2, 0, 1)  # [K, H, S]

    K, H, S = data.shape  # 统一的布局：data[k, h, s]

    if separate_plots:
        # 每个 (head, group/dim) 一张图
        for h in tqdm(range(H), desc="heads"):
            for g in range(S):
                plt.figure()
                plt.plot(x_axis, data[:, h, g])
                plt.xlabel("index (T or freq)")
                plt.ylabel("value")
                plt.title(f"{name} | head={h}, {group_labels[g]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"{name}_head{h}_group{g}.png"),
                    dpi=200,
                )
                plt.close()
    else:
        # 每个 head 一张图，里面画多个 group/dim 曲线
        for h in tqdm(range(H), desc="heads"):
            plt.figure()
            for g in range(S):
                plt.plot(x_axis, data[:, h, g], label=group_labels[g])
            plt.xlabel("index (T or freq)")
            plt.ylabel("value")
            plt.title(f"{name} | head={h}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"{name}_head{h}_groups.png"),
                dpi=200,
            )
            plt.close()


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


def spectral_distill_over_L(
    student: torch.Tensor,   # [B, L, H, D]  (path_attn_scores 映射/reshape到该形状)
    teacher: torch.Tensor,   # [B, L, H, D]  (wavelet_scores 映射/reshape到该形状)
    distill_teacher:str,    layer_idx: int,
    name,
    start_idx,
    out_dir,
    *, tau: float = 1.0,
    w_band: torch.Tensor | None = None,  # [K] 可选频带权重
    lambda_mse: float = 1.0,
    lambda_kl: float = 0.5,
    lambda_cos: float = 0.0,             # 需要时再开
):
    with torch.no_grad():
        A_t, A_t_log = spectrum_over_T_multi(teacher)   # [B,Q,K,H,D]  teacher不反传
    A_s, A_s_log = spectrum_over_T_multi(student)       # [B,Q,K,H,D]
    A_t_scale = aggregate_spectrum_by_scale(A_t, group_size=8, distill_teacher=distill_teacher)  # [H, S, K]
    A_s_scale = aggregate_spectrum_by_scale(A_s, group_size=8, distill_teacher=distill_teacher)
    freq_out_dir = out_dir + 'spectral_analysis_logs'
    os.makedirs(freq_out_dir, exist_ok=True)
    os.makedirs(f'{freq_out_dir}/var_analysis', exist_ok=True)
############ var analysis #################
    H, S, K = A_t_scale.shape
    var_t = A_t_scale.var(dim=-1, unbiased=False)   # teacher: [H, S]
    var_s = A_s_scale.var(dim=-1, unbiased=False)   # student: [H, S]

    # 2. 如果你想快速浏览每个 head / group 的方差对比：
    out_path = f"{freq_out_dir}/var_analysis/layer_{layer_idx}_start_{start_idx}_var_stats.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for h in tqdm(range(H), desc="heads"):
            for s in range(S):
                vt = var_t[h, s].item()
                vs = var_s[h, s].item()
                line = (
                    f"head {h:2d}, group {s:2d}  |  "
                    f"teacher var = {vt:.4e},  student var = {vs:.4e}\n"
                )
                f.write(line)

    # print(f"saved to {out_path}")
    # pdb.set_trace()
############ var analysis #################
    os.makedirs(f'{freq_out_dir}/spectrum_domain_plots', exist_ok=True)
    # stats_t, stats_s = analyze_teacher_student_groups(
    #     A_t_scale=A_t_scale,
    #     A_s_scale=A_s_scale,
    #     layer_idx=layer_idx,
    #     outdir_base=f"{freq_out_dir}/plots_group_similarity",
    #     start_idx=start_idx,
    # )
    if distill_teacher == 'wavelet':
        plot_out_head_dim_groups_or_grouped(A_t[:, 0, ...], f'{freq_out_dir}/spectrum_domain_plots', name+'_teacher', distill_teacher=distill_teacher)
    plot_out_head_dim_groups_or_grouped(A_s[:, 0, ...], f'{freq_out_dir}/spectrum_domain_plots', name+'_student', distill_teacher=distill_teacher)
    # t_mean, s_mean, kl_mat, row_ind, col_ind = match_heads_by_kl_over_S(
    #     A_t_scale, A_s_scale
    # )

    # plot_matched_heads_over_freq(
    #     t_mean,
    #     s_mean,
    #     row_ind,
    #     col_ind,
    #     save_path=f"freq_analysis_logs/layer{layer_idx}_matched_heads_freq.png",
    #     title_prefix=f"Layer {layer_idx}"
    # )
    # pdb.set_trace()
    # if layer_idx == 0:
        # print("layer0 teacher power sum:", A_t_scale.abs().sum())
        # print("layer0 student power sum:", A_s_scale.abs().sum())
    # pdb.set_trace()
    stats = spectrum_stats_teacher_student(A_t_scale, A_s_scale, low_ratio=0.25)
    if start_idx not in all_stats_per_layer:
        all_stats_per_layer[start_idx] = [stats]
    else:
        all_stats_per_layer[start_idx].append(stats)
    
    
    plot_layer_dashboard(stats, freq_out_dir, name)
    if layer_idx == 11:
        summarize_over_layers(all_stats_per_layer[start_idx], freq_out_dir, name)
        plot_kl_cos_heatmap_over_layers(all_stats_per_layer[start_idx], freq_out_dir, name)
        all_stats_per_layer[start_idx].clear()
    # 频带加权（可选）
    if w_band is not None:
        # w_band: [K] -> [1,1,1,K]
        w = w_band.to(A_s).view(1,1,1,-1)
    else:
        w = 1.0

    # 1) 对数幅值 MSE
    loss_spec_mse = ((A_s_log - A_t_log)**2 * w).mean()
    return loss_spec_mse

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
def path_attn_multi_query_elementwise(Q_sel, K, W, beta, offsets=(1)):
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
from pathlib import Path
from rotary_embedding_torch import RotaryEmbedding
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
        self.config=    config
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
        if self.config.distill_teacher == 'rotary':
            self.rotary_emb = RotaryEmbedding(dim=64)
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
            # gamma_stats = compute_gamma_stats(w=w, beta=beta)
            # plot_gamma_lambda_hist_by_head(gamma_stats['gamma'], gamma_stats['lambda_dir'], ratio_contractive=gamma_stats['ratio_contractive'], title_prefix=f"wavelet_distill_layer{self.layer_idx}", outdir='gamma_lambda_hist')
            o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)
            # if (self.layer_idx < 2) and self.training:
            offsets = (1, 8, 16, 32)
            # idx = [k.size(1) - o for o in offsets]       # 绝对下标
            # Q_sel = q[:, idx, :, :]
            print(f'layer{self.layer_idx} analysis!!!!')
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
                    spectral_teacher_scores = (q[:, -1:, ...] * k)
                    norm = spectral_teacher_scores.norm(dim=1, keepdim=True) + 1e-12
                    spectral_teacher_scores = spectral_teacher_scores / norm            
                elif self.config.distill_teacher == 'shrink_w_shuffle':
                    spectral_teacher_scores = (q[:, -1:, ...] * k)
                    norm = spectral_teacher_scores.norm(dim=1, keepdim=True) + 1e-12
                    spectral_teacher_scores = spectral_teacher_scores / norm
                    spectral_teacher_scores = make_randomized_teacher_T(spectral_teacher_scores)
                else:
                    raise ValueError(f"Unknown distill_teacher: {self.config.distill_teacher}")
            # with torch.no_grad():
            #     if self.config.distill_teacher == 'rotary':
            #         dim_wise_scores = self.rotary_emb.rotate_queries_or_keys(q.permute(0, 2, 1, 3).contiguous()) * self.rotary_emb.rotate_queries_or_keys(k.permute(0, 2, 1, 3).contiguous())
            #         spectral_teacher_scores = dim_wise_scores.permute(0, 2, 1, 3)
            #     elif self.config.distill_teacher == 'wavelet':
            #         spectral_teacher_scores = compute_wavelet_scores_batched(q[:, -1, ...], k, wavelet_decay_table[:, -1, :])
            #     else:
            #         raise ValueError(f"Unknown distill_teacher: {self.config.distill_teacher}")            
            # with torch.no_grad():
            #     # 计算 wavelet 分数
            #     # wavelet_scores = compute_wavelet_scores_multi_causal(Q_sel, k, wavelet_decay_table[:, -1, :], query_indices=idx, table_direction="near_to_far")
            #     wavelet_scores = compute_wavelet_scores_batched(q[:, -1, ...], k, wavelet_decay_table[:, -1, :])
            
            path_attn_scores = compute_path_scores_batched_last_q(q[:, -1, ...], k, w, beta)
            # diff = path_attn_scores[..., 1:] - path_attn_scores[..., :-1]
            # E_diff = torch.mean(diff**2)
            # print(E_diff)
            # pdb.set_trace()
            # spectral_teacher_scores = F.softmax(spectral_teacher_scores, dim=-3)
            # path_attn_scores = F.softmax(path_attn_scores, dim=-3)
            num_in_group = 128
            group_num = q.size(1) // num_in_group
            if self.config.block_size < num_in_group:
                group_num = 1
                num_in_group = self.config.block_size
            out_dir = Path(self.config.model_name_or_path).name + '_' + Path(self.config.model_name_or_path).parent.name + '_'
            # softmax_out_dir = 'softmax_' + out_dir
            temporal_out_dir = out_dir + 'temporal_domain_plots'
            # softxmax_temporal_out_dir = 'softmax_'+temporal_out_dir
            # softmax_path_attn_scores = F.softmax(path_attn_scores, dim=-3)
            # softmax_teacher_scores = F.softmax(spectral_teacher_scores, dim=-3)
            # for group in range(group_num):
            #     start_idx = group * num_in_group
            #     end_idx = (group + 1) * num_in_group
            #     group_path_attn_scores = path_attn_scores[:, start_idx:end_idx, ...]
            #     group_teacher_scores = spectral_teacher_scores[:, start_idx:end_idx, ...]
            #     softmax_group_path_attn_scores = softmax_path_attn_scores[:, start_idx:end_idx, ...]
            #     softmax_group_teacher_scores = softmax_teacher_scores[:, start_idx:end_idx, ...]
            #     plot_out_head_dim_groups_or_grouped(group_path_attn_scores, f'{temporal_out_dir}', name=f"layer{self.layer_idx}_student_{start_idx}_to_{end_idx}", distill_teacher=self.config.distill_teacher)
            #     plot_out_head_dim_groups_or_grouped(softmax_group_path_attn_scores, f'{softxmax_temporal_out_dir}', name=f"layer{self.layer_idx}_student_{start_idx}_to_{end_idx}", distill_teacher=self.config.distill_teacher)
            #     if self.config.distill_teacher == 'wavelet':
            #         plot_out_head_dim_groups_or_grouped(softmax_group_teacher_scores, f'{softxmax_temporal_out_dir}', name=f"layer{self.layer_idx}_teacher_{start_idx}_to_{end_idx}", distill_teacher=self.config.distill_teacher)
            #         plot_out_head_dim_groups_or_grouped(group_teacher_scores, f'{temporal_out_dir}', name=f"layer{self.layer_idx}_teacher_{start_idx}_to_{end_idx}", distill_teacher=self.config.distill_teacher)
                
            #     dis_loss = spectral_distill_over_L(group_path_attn_scores.unsqueeze(1), group_teacher_scores.unsqueeze(1), distill_teacher=self.config.distill_teacher, layer_idx=self.layer_idx, name = f"layer{self.layer_idx}_{start_idx}_to_{end_idx}", start_idx=start_idx, out_dir=out_dir)
            #     _ = spectral_distill_over_L(softmax_group_path_attn_scores.unsqueeze(1), softmax_group_teacher_scores.unsqueeze(1), distill_teacher=self.config.distill_teacher, layer_idx=self.layer_idx, name = f"layer{self.layer_idx}_{start_idx}_to_{end_idx}", start_idx=start_idx, out_dir=softmax_out_dir)
            plot_out_head_dim_groups_or_grouped(path_attn_scores, f'{temporal_out_dir}', name=f"layer{self.layer_idx}_student_full", distill_teacher=self.config.distill_teacher)
            # plot_out_head_dim_groups_or_grouped(softmax_path_attn_scores, f'{softxmax_temporal_out_dir}', name=f"layer{self.layer_idx}_student_full", distill_teacher=self.config.distill_teacher)
            if self.config.distill_teacher == 'wavelet':
                plot_out_head_dim_groups_or_grouped(spectral_teacher_scores, f'{temporal_out_dir}', name=f"layer{self.layer_idx}_teacher_full", distill_teacher=self.config.distill_teacher)
                # plot_out_head_dim_groups_or_grouped(softmax_teacher_scores, f'{softxmax_temporal_out_dir}', name=f"layer{self.layer_idx}_teacher_full", distill_teacher=self.config.distill_teacher)
            dis_loss = spectral_distill_over_L(path_attn_scores.unsqueeze(1), spectral_teacher_scores.unsqueeze(1), distill_teacher=self.config.distill_teacher, layer_idx=self.layer_idx, name = f"layer{self.layer_idx}_full", start_idx=-1, out_dir=out_dir)
            # _ = spectral_distill_over_L(softmax_path_attn_scores.unsqueeze(1), softmax_teacher_scores.unsqueeze(1), distill_teacher=self.config.distill_teacher, layer_idx=self.layer_idx, name = f"layer{self.layer_idx}_full", start_idx=-1, out_dir=softmax_out_dir)
            
            # else:
            #     dis_loss = torch.tensor(0.0, device=q.device, dtype=q.dtype)

            if self.layer_idx == 11:
                os._exit(0)
            o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
            o = self.o_proj(o)
            return o, None, past_key_values, 0

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