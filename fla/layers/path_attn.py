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
def compute_wavelet_score_single(q_j, k_i, wavelet_decay_list, i_idx, j_idx, *, sqrt_d_scale=True):
    H, D = q_j.shape
    rel = j_idx - i_idx                        # 假设 0<=rel<R
    d = wavelet_decay_list[:, rel].to(q_j)     # [D]
    q_w = q_j * d                              # [H,D]
    score = torch.einsum('hd,hd->', q_w, k_i)  # 标量

    return score / math.sqrt(D)
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

    score_h = (q_j * x).sum(dim=-1)     # 每个 head 的标量
    score = score_h.sum()               # 跨 head 求和 -> 标量
    return score/math.sqrt(D)
import random
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
            if self.wavelet_baseline_use:
                qk = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2))
                rel = torch.einsum("blhd,dln->blhn", q, wavelet_decay_table)
                rel= rel.transpose(1, 2)
                wavelet_bias = (qk + rel) / torch.full(
                    [], self.head_dim ** 0.5, dtype=q.dtype, device=q.device
                )
                mask_value = torch.finfo(wavelet_bias.dtype).min
                # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                mask_value = torch.full([], mask_value, dtype=wavelet_bias.dtype, device=wavelet_bias.device)
                wavelet_bias = torch.where(build_causal_mask(rel.size(-1),rel.size(-1), device='cuda'), wavelet_bias.to(wavelet_bias.dtype), mask_value)
                wavelet_bias = nn.functional.softmax(wavelet_bias, dim=-1)
                wavelet_bias = self.attn_dropout(wavelet_bias)
                attn_output = torch.matmul(wavelet_bias, v.transpose(1, 2))
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
            if self.layer_idx == 5 and self.training:
                i_idx, j_idx, deltas = sample_index_pairs(self.config.block_size, num_samples=self.config.sample_num, geom_p=geom_p)
                for idx in range(self.config.sample_num):
                    wavelet_scores = torch.empty(self.config.sample_num, device=q.device, dtype=q.dtype)
                    path_attn_scores = torch.empty(self.config.sample_num, device=q.device, dtype=q.dtype)
                    i = i_idx[idx]
                    j = j_idx[idx]
                    delta = deltas[idx]
                    b_idx = random.randint(0, 15)
                    path_attn_scores[idx] = compute_path_score_single(q[b_idx, j], k[b_idx, i], w[b_idx], beta[b_idx], i.item(), j.item())
                    wavelet_scores[idx] = compute_wavelet_score_single(q[b_idx, j], k[b_idx, i], wavelet_decay_table[:, -1, :], i.item(), j.item())
                dis_loss = F.mse_loss(path_attn_scores, wavelet_scores)
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
        # pdb.set_trace()
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