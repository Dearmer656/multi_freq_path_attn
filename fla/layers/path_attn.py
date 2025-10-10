# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import logging

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

logger = logging.get_logger(__name__)

import pdb
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
    ):
        super().__init__()

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

        # w 分支输出扩到 H*R*d（保持你原有低秩/全连接两种实现）
        out_w = self.kv_dim * self.r
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w, bias=False)
            )
        else:
            self.w_proj = nn.Linear(self.hidden_size, out_w, bias=False)

        # Q/K 归一保持和原版一致
        if use_qk_norm:
            self.maybe_q_norm = RMSNorm(self.hidden_size)
            self.maybe_k_norm = RMSNorm(self.kv_dim)
        else:
            self.maybe_q_norm = nn.Identity()
            self.maybe_k_norm = nn.Identity()

        # depthwise short conv 的通道也扩到 H*R*d
        self.use_w_shortconv = use_w_shortconv
        if use_w_shortconv:
            self.w_conv1d = ShortConvolution(
                hidden_size=out_w, kernel_size=conv_size, bias=conv_bias, activation='silu'
            )

        # 每个 (head, rank) 一个 beta
        self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.r, bias=True)

        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        # 输出层输入维度改为 H*R*d（= hidden_size * R），映射回 hidden_size
        self.o_proj = nn.Linear(self.hidden_size * self.r, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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

        q = self.q_proj(hidden_states)           # [B,T,HQ*d]
        k = self.k_proj(hidden_states)           # [B,T,H*d]   (H = num_kv_heads)
        v = self.v_proj(hidden_states)           # [B,T,H*d]
        w = self.w_proj(hidden_states)           # [B,T,H*R*d]
        beta = self.bt_proj(hidden_states).sigmoid() * 2  # [B,T,H*R]
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None

        q, k = self.maybe_q_norm(q), self.maybe_k_norm(k)
        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert not (cu_seqlens is not None and attention_mask is not None), (
            "cu_seqlens should not be provided when attention_mask is not None"
        )

        # 训练路径（和你给的版本保持一致）
        if attention_mask is None:
            assert use_cache is False, "use_cache should be False in training"

            # w 分支卷积（对拼接后的 H*R*d）
            if self.use_w_shortconv:
                w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)

            # 整理到多头格式
            q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_dim)                     # HQ 可能 != H
            k = rearrange(k, 'b t (h d) -> b t h d',  d=self.head_dim)                     # H
            v = rearrange(v, 'b t (h d) -> b t h d',  d=self.head_dim)                     # H
            W = rearrange(w, 'b t (h r d) -> b t h r d', h=self.num_kv_heads, r=self.r, d=self.head_dim)  # [B,T,H,R,d]
            W = l2_norm(W)  # 对最后一维 d 做 L2 归一

            # 把 rank 维并到头维：q/k/v 重复 R 倍；w/beta reshape 合并
            q = q.repeat_interleave(self.r, dim=2)                                         # [B,T,HQ*R,d]
            k = k.repeat_interleave(self.r, dim=2)                                         # [B,T,H*R,d]
            v = v.repeat_interleave(self.r, dim=2)                                         # [B,T,H*R,d]
            w = rearrange(W, 'b t h r d -> b t (h r) d')                                   # [B,T,H*R,d]
            beta = rearrange(beta, 'b t (h r) -> b t (h r)', h=self.num_kv_heads, r=self.r)

            # g（若开启）也在 HQ 维做 repeat_interleave(R)
            if g is not None:
                g = rearrange(g, 'b t hq -> b t hq 1').repeat(1, 1, self.r, 1).view(g.shape[0], g.shape[1], -1)

            # 直接调原内核（接口不变）
            o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)

            # 合并回隐维，再做输出投影（H*R*d → hidden_size）
            o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
            o = self.o_proj(o)
            return o, None, past_key_values

        # 你给的这份类没有实现 mask≠None 的预填充/解码分支；保持不变
        # 如需推理/缓存，请把原版的 Prefill/Decode 分支也做同样的 rank 展开（和上面训练分支一致的思路）。
        # 这里直接走训练同款（仅为完整性，通常不会在 mask!=None 时进入这里）：
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
        beta = rearrange(beta, 'b t (h r) -> b t (h r)', h=self.num_kv_heads, r=self.r)
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
        
        self.phi_raw = nn.Parameter(torch.zeros(1,1,Hf,self.r))        
        nn.init.constant_(self.bt_proj.bias, -2.0)  # β 小一点
        self.use_wavelet_beta = use_wavelet_beta           # 开关：想先关就置 False
        self.wavelet_mode = wavelet_mode        # "additive" 或 "softmix"
        self.beta_amp = nn.Parameter(torch.zeros(1,1,Hf,self.r))
        # 幅度 a（从 0 起，不扰动基线），σ 的安全参数化，给个下限避免太尖
        self.ricker_amp   = nn.Parameter(torch.zeros(1,1,Hf,self.r))              # a
        self.ricker_log_s = nn.Parameter(torch.full((1,1,Hf,self.r), -0.7))       # softplus 后再加下限
        self.sigma_min = 8.0                                                       # token 级下限，可按需要调

        # 若选择 softmix，这个是可学习的混合系数 λ 的 logit
        self.mix_logit = nn.Parameter(torch.tensor(0.0)) if True else None  # 若只用 additive 可删
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
            # 末位对齐：pos ∈ [-T+1, ..., 0]，在 0 处峰值
            pos_end = torch.arange(-T+1, 1, device=hidden_states.device).view(1, T, 1, 1).to(hidden_states.dtype)  # [1,T,1,1]

            # σ 安全参数化
            sigma = F.softplus(self.ricker_log_s) + self.sigma_min                     # [1,1,Hf,r]

            # === 将 get_omega / get_phi 应用于 wavelet 的时间轴 ===
            omega_w = self.get_omega()                                                 # [1,1,Hf,r]
            phi_w   = self.get_phi()                                                   # [1,1,Hf,r]

            # 把相位/频率换算到“token 轴”的仿射参数：
            # scale: 相对 omega_base 的无量纲缩放；shift_tokens: 相位对应的 token 级平移
            scale = (omega_w / self.omega_base).to(hidden_states.dtype)                # [1,1,Hf,r]
            shift_tokens = (phi_w / (omega_w + 1e-6)).to(hidden_states.dtype)          # [1,1,Hf,r]

            # 广播到 [1,T,Hf,r]
            scale = scale.expand(1, T, scale.size(2), scale.size(3))
            shift_tokens = shift_tokens.expand_as(scale)

            # 先平移再缩放：t_affine = scale * (pos_end - shift_tokens)
            t_affine = scale * (pos_end - shift_tokens)                                # [1,T,Hf,r]

            # 归一化距离 τ，用于 Ricker
            tau = t_affine / (sigma + 1e-6)                                            # [1,T,Hf,r]

            psi = (1.0 - tau**2) * torch.exp(-0.5 * tau**2)                            # [1,T,Hf,r]
            psi = psi - psi.mean(dim=1, keepdim=True)
            wave = (self.ricker_amp * psi).to(beta_logits.dtype)                       # [1,T,Hf,r]
            pdb.set_trace()
            # 展到 [B,T,H,R] 再拉平
            Hf = wave.size(2)
            if Hf == 1:
                wave = wave.expand(B, T, self.num_kv_heads, self.r)                    # [B,T,H,r]
            else:
                wave = wave.expand(B, T, Hf, self.r)
            wave = rearrange(wave, 'b t h r -> b t (h r)')                             # [B,T,H*R]

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