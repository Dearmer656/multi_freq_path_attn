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