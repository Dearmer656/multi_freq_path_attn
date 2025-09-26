# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.path_attn.parallel import parallel_path_attn

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)

import pdb

class PaTHAttention(nn.Module):
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
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.r = num_harmonics
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
        out_w = self.kv_dim * (2 * self.r)
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, out_w, bias=False),
            )
        else:
            self.w_proj = nn.Linear(self.hidden_size, out_w, bias=False)

        # TODO: per head norm?
        if use_qk_norm:
            self.maybe_q_norm = RMSNorm(self.hidden_size)
            self.maybe_k_norm = RMSNorm(self.kv_dim)
        else:
            self.maybe_q_norm = nn.Identity()
            self.maybe_k_norm = nn.Identity()

        if use_w_shortconv:
            # 卷积的通道数也改成 2R 倍
            self.w_conv1d = ShortConvolution(hidden_size=out_w, kernel_size=conv_size,
                                             bias=conv_bias, activation='silu')
        self.use_w_shortconv = use_w_shortconv
        # β: 每个频带一个 β 门（范围 0~2）
        self.bt_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.r, bias=True)

        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        self.o_proj = nn.Linear(self.hidden_size * self.r, self.hidden_size, bias=False)
        Hf = 1 if share_freq_across_heads else self.num_kv_heads
        self.omega = nn.Parameter(0.01 * torch.ones(1, 1, Hf, self.r))  # 小值初始化更稳
        self.phi   = nn.Parameter(torch.zeros(1, 1, Hf, self.r))
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
        
        batch_size, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        w = self.w_proj(hidden_states)
        beta = self.bt_proj(hidden_states).sigmoid() * 2  # allowing negative eigenvalues
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None
        # print(torch.max(q), torch.max(k))
        # print()
        q, k = self.maybe_q_norm(q), self.maybe_k_norm(k)
        # print(torch.max(q), torch.max(k))
        # print(hidden_states, 'hidden_states in PaTHAttention.')
        # print(beta, 'beta in PaTHAttention.')
        
        # pdb.set_trace()
        # print('!!!!!!!!!!!!!!!')
        cu_seqlens = kwargs.get('cu_seqlens', None)
        assert not (cu_seqlens is not None and attention_mask is not None), (
            "cu_seqlens should not be provided when attention_mask is not None"
        )
        # Training
        if attention_mask is None:
            B, T, _ = hidden_states.size()
            pos = torch.arange(T, device=hidden_states.device, dtype=hidden_states.dtype)[None, :, None, None]  # [1,T,1,1]
            omega = self.omega if not self.share_freq_across_heads else self.omega.expand(1,1,self.num_kv_heads,self.r)
            phi   = self.phi   if not self.share_freq_across_heads else self.phi.expand(1,1,self.num_kv_heads,self.r)
            theta = omega * pos + phi                           # [1,T,H,R]
            c, s = torch.cos(theta), torch.sin(theta)
            assert use_cache is False, "use_cache should be False in training"
            w_raw = self.w_proj(hidden_states)
            if self.use_w_shortconv:
                w_raw, _ = self.w_conv1d(w_raw, cache=None, output_final_state=False, cu_seqlens=kwargs.get('cu_seqlens', None))
          
            # [B,T,H,2R,d]
            w2 = rearrange(w_raw, 'b t (h rr d) -> b t h rr d', h=self.num_kv_heads, rr=2*self.r, d=self.head_dim)
            A = w2[:, :, :, 0::2, :]                            # [B,T,H,R,d]
            B_ = w2[:, :, :, 1::2, :]                           # [B,T,H,R,d]
            # 广播到 [B,T,H,R,d]
            c = c.to(A.dtype).expand(B, T, self.num_kv_heads, self.r)[..., None]
            s = s.to(A.dtype).expand(B, T, self.num_kv_heads, self.r)[..., None]
            W = A * c + B_ * s                                  # [B,T,H,R,d]
            W = l2_norm(W)                          
            # 重排所有张量到多头格式
            beta = self.bt_proj(hidden_states).sigmoid() * 2    # [B,T,H*R]
            beta = rearrange(beta, 'b t (h r) -> b t h r', h=self.num_kv_heads, r=self.r)
            q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_dim)
            k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_dim)
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_dim)

            # 展开：repeat_interleave 比较直接
            q = q.repeat_interleave(self.r, dim=2)              # [B,T,H*R,d]
            k = k.repeat_interleave(self.r, dim=2)
            v = v.repeat_interleave(self.r, dim=2)
            w = rearrange(W, 'b t h r d -> b t (h r) d')        # [B,T,H*R,d]
            beta = rearrange(beta, 'b t h r -> b t (h r)')      # [B,T,H*R]

            if g is not None:
                g = rearrange(g, 'b t h -> b t h 1').repeat(1,1,1*self.r,1).squeeze(-1)

            # 5) 调 parallel_path_attn（签名不变）
            o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g,
                                    cu_seqlens=kwargs.get('cu_seqlens', None), use_cache=False)


        # Prefilling or decoding
        else:
            assert self.training is False, "attention mask is not supported in training. Please use variable length input."
            try:
                last_state = past_key_values[self.layer_idx]
            except KeyError:
                last_state = None
            # Decoding
            if last_state is not None:
                if g is not None:
                    past_k, past_v, past_g = last_state['attn_state']
                else:
                    past_k, past_v = last_state['attn_state']
                w_conv_state = last_state['conv_state']
                past_k = rearrange(past_k, '... (h d) -> ... h d', d=self.head_dim)
                if self.use_w_shortconv:
                    w, w_conv_state = self.w_conv1d(w, cache=w_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
                w = l2_norm(w)

                @torch.compile
                def rank_one_update(k, w, beta):
                    original_dtype = k.dtype
                    k = k.float()
                    w = w.float()
                    beta = beta.float()
                    k = k - beta[..., None].float() * (k * w).sum(-1, keepdim=True) * w
                    return k.to(original_dtype)

                past_k = rank_one_update(past_k, w, beta)
                past_k = rearrange(past_k, '... h d -> ... (h d)')
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
                g = torch.cat([past_g, g], dim=1) if g is not None else None
                past_key_values[self.layer_idx]['attn_state'] = (k, v, g) if g is not None else (k, v)
                past_key_values.update(
                    conv_state=w_conv_state,
                    layer_idx=self.layer_idx,
                    offset=q_len
                )
                if g is not None:
                    q, (k, v, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                        q, (k, v, g), attention_mask, q_len, keepdim=True)
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                else:
                    q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                        q, (k, v), attention_mask, q_len, keepdim=True)
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                _, cu_seqlens = cu_seqlens
                q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
                assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                o = attn_decoding_one_step(q, k, v, g, cu_seqlens=cu_seqlens, do_gate_scale=True)  # reduced to fox's decoding
            # Prefilling
            else:
                v_cache = v.clone()
                g_cache = g.clone() if g is not None else None
                if g is None:
                    q, (k, v, w, beta), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                        q, (k, v, w, beta), attention_mask, q_len, keepdim=True)
                else:
                    q, (k, v, w, beta, g), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                        q, (k, v, w, beta, g), attention_mask, q_len, keepdim=True)
                max_seqlen_q, max_seqlen_k = max_seq_lens
                assert max_seqlen_q == max_seqlen_k, "max_seqlen_q should be equal to max_seqlen_k in prefilling"
                _, cu_seqlens = cu_seqlens
                if self.use_w_shortconv:
                    w, w_conv_state = self.w_conv1d(w, cache=None, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                else:
                    w_conv_state = None
                q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
                w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
                w = l2_norm(w)
                o, k_cache = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g,
                                                cu_seqlens=cu_seqlens, use_cache=use_cache)
                if use_cache:
                    k_cache = pad_input(k_cache.squeeze(0), indices_q, batch_size, q_len)
                    k_cache = rearrange(k_cache, '... h d -> ... (h d)')
                    past_key_values.update(
                        attn_state=(k_cache, v_cache, g_cache) if g_cache is not None else (k_cache, v_cache),
                        conv_state=w_conv_state,
                        layer_idx=self.layer_idx,
                        offset=q_len
                    )
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        o = rearrange(o, 'b t (h r) d -> b t (h r d)', r=self.r)
        ##!!!
        o = self.o_proj(o)
        return o, None, past_key_values
