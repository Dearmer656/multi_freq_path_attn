
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from fla.ops.utils import prepare_chunk_indices
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
@triton.heuristics({
    "USE_G": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_fwd_kernel(
    q, k, v, w, beta, g_cumsum, o, A, L, M, w2, q_new, k_new,
    scale,
    indices,   # varlen helper: [NT, 2] of (n, ib)
    offsets,   # varlen helper: cu_seqlens
    T,
    H: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    # ==== 新增：传入每个 (head×rank) 的一维 tail 表，形状 [H*R, BT] ====
    decay_table,                 # tensor pointer; 若禁用也需占位传空张量
    USE_DECAY: tl.constexpr,     # 是否启用衰减（编译期开关）
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    # ----- 变量长度：解析本 CTA 对应的样本与块 -----
    if IS_VARLEN:
        i_n = tl.load(indices + i_t * 2 + 0).to(tl.int32)
        i_t = tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(offsets + i_n).to(tl.int32)
        eos = tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
        eos = bos + T

    sm_scale = scale * 1.44269504

    # ----- base address 偏移 -----
    A      += (bos * H  + i_h ) * BT
    q      += (bos * HQ + i_hq) * K
    q_new  += (bos * HQ + i_hq) * K
    k      += (bos * H  + i_h ) * K
    k_new  += (bos * H  + i_h ) * K
    w2     += (bos * H  + i_h ) * K
    w      += (bos * H  + i_h ) * K
    v      += (bos * H  + i_h ) * V
    o      += (bos * HQ + i_hq) * V
    beta   += (bos * H  + i_h )
    if USE_G:
        g_cumsum += (bos * HQ + i_hq)
    L      += (bos * HQ + i_hq)
    M      += (bos * HQ + i_hq)

    # ----- load 本块 -----
    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q  = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_v  = tl.load(p_v, boundary_check=(0, 1))
    b_w  = tl.load(p_w, boundary_check=(0, 1))
    p_T  = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T  = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]

    # ----- 本 tile 实际长度（varlen 最后一块可能不足 BT）-----
    T_tile = tl.minimum(BT, T - i_t * BT)

    # ----- β 行向量（来源步 s 的缩放）-----
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))                  # [BT]
    b_w_beta = (b_w * b_beta[:, None])                             # [BT,K]（保留给 w2）

    # ----- q·w^T（无 β/κ），构造 qwT -----
    b_qw  = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)  # [BT,BT]
    b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)                              # [BT,BT]

    # ----- 关键：构造 κ(i-s)，并形成 coeff = β_s * κ(i-s) -----
    if USE_DECAY:
        # delta 仅依赖块内相对距离：Δ∈[0, T_tile-1]
        delta = (o_i[:, None] - o_i[None, :]).to(tl.int32)              # [BT,BT]
        delta = tl.maximum(delta, 0)
        delta = tl.minimum(delta, T_tile - 1)

        # decay_table: [H*R, BT]（每个 head×rank 一行，长度=BT 的 tail）
        i_hr = i_h * G + (i_hq % G)
        row_ptr = decay_table + i_hr * BT                                # 行起点
        kappa = tl.load(row_ptr + delta)            # [BT,BT]
        beta_col = b_beta[None, :]                                       # [1,BT]
        coeff = (beta_col * kappa).to(b_q.dtype)                         # [BT,BT]
    else:
        # 退化：仅列向 β（按列广播）
        coeff = b_beta[None, :].to(b_q.dtype)                            # [BT,BT]

    # ----- b_wbk：原本是 dot(b_w_beta, b_kt)，现在 = (coeff * base) -----
    b_wbk_base = tl.dot(b_w, b_kt).to(b_q.dtype)                         # [BT,BT]
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], coeff * b_wbk_base, 0).to(b_q.dtype)

    # ----- A = qk^T - (qwT)(wbk) -----
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

    # ----- q ← q − ((qwT ⊙ coeff) @ w) -----
    weighted = (b_qwT * coeff).to(b_w.dtype)                             # [BT,BT]
    b_q = b_q - tl.dot(weighted, b_w).to(b_q.dtype)
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    # ----- 仅首个 query-rank 写 w2 / k_new（与你原逻辑一致）-----
    if (i_hq % G) == 0:
        b_Twb = tl.dot(b_T, b_w_beta).to(b_w.dtype)                      # 这里保持仅 β，不乘 κ
        p_w2 = tl.make_block_ptr(w2, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        tl.store(p_w2, b_Twb, boundary_check=(0, 1))

        b_T_wbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w), b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    # ----- g 门（如启用）-----
    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A = b_A + (b_g[:, None] - b_g[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))

    # ----- softmax & write -----
    b_qkT = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT, 1)
    b_qkT = tl.math.exp2(b_qkT - m_i[:, None])
    l_i = tl.sum(b_qkT, 1)
    b_o = tl.dot(b_qkT.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (V*HQ, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = tl.make_block_ptr(L, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_m = tl.make_block_ptr(M, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), boundary_check=(0,))
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), boundary_check=(0,))
# @triton.jit(do_not_specialize=['T'])
# def intra_chunk_preprocess_fwd_kernel(
#     q,
#     k,
#     v,
#     w,
#     beta,
#     g_cumsum,
#     o,
#     A,
#     L,
#     M,
#     w2,
#     q_new,
#     k_new,
#     scale,
#     indices,  # varlen helper
#     offsets,  # varlen helper
#     T,
#     H: tl.constexpr,
#     G: tl.constexpr,
#     HQ: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     BT: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     USE_G: tl.constexpr,
# ):
#     i_t, i_nh = tl.program_id(0), tl.program_id(1)
#     i_n, i_hq = i_nh // HQ, i_nh % HQ
#     i_h = i_hq // G

#     if IS_VARLEN:
#         i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
#         T = eos - bos
#     else:
#         bos, eos = i_n * T, i_n * T + T

#     sm_scale = scale * 1.44269504
#     # offset calculations
#     A += (bos*H + i_h) * BT
#     q += (bos*HQ + i_hq) * K
#     q_new += (bos*HQ + i_hq) * K
#     k += (bos*H + i_h) * K
#     k_new += (bos*H + i_h) * K
#     w2 += (bos*H + i_h) * K
#     w += (bos*H + i_h) * K
#     v += (bos*H + i_h) * V
#     o += (bos*HQ + i_hq) * V
#     beta += (bos*H + i_h)
#     if USE_G:
#         g_cumsum += (bos*HQ + i_hq)
#     L += (bos*HQ + i_hq)
#     M += (bos*HQ + i_hq)

#     p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#     p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
#     p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#     p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
#     b_q = tl.load(p_q, boundary_check=(0, 1))
#     b_kt = tl.load(p_k, boundary_check=(0, 1))
#     b_v = tl.load(p_v, boundary_check=(0, 1))
#     b_w = tl.load(p_w, boundary_check=(0, 1))
#     p_T = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
#     b_T = tl.load(p_T, boundary_check=(0, 1))

#     o_i = tl.arange(0, BT)
#     m_t = o_i[:, None] >= o_i[None, :]
#     p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
#     b_beta = tl.load(p_beta, boundary_check=(0, ))
#     b_w_beta = (b_w * b_beta[:, None])

#     b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
#     b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)
#     b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, b_kt), 0).to(b_q.dtype)
#     b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

#     b_q = b_q - tl.dot(b_qwT, b_w_beta)
#     p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, K), (1, 0))
#     tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

#     if i_hq % G == 0:
#         b_Twb = tl.dot(b_T, b_w_beta).to(b_w.dtype)
#         p_w2 = tl.make_block_ptr(w2, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#         tl.store(p_w2, b_Twb, boundary_check=(0, 1))
#         b_T_wbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
#         p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
#         tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w), b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

#     if USE_G:
#         p_g_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
#         b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0, ))
#         b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
#         b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

#     b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
#     m_i = tl.max(b_qkT_softmax, 1)
#     b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
#     l_i = tl.sum(b_qkT_softmax, 1)
#     b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
#     p_o = tl.make_block_ptr(o, (T, V), (V*HQ, 1), (i_t * BT, 0), (BT, BV), (1, 0))
#     tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
#     p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
#     p_m = tl.make_block_ptr(M, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
#     tl.store(p_m, m_i.to(p_m.dtype.element_ty), boundary_check=(0,))
#     tl.store(p_l, l_i.to(p_l.dtype.element_ty), boundary_check=(0,))


def intra_chunk_preprocess_fwd_fn(q, k, v, w, beta, g_cumsum, A, scale, BT, cu_seqlens, decay_table):
    USE_DECAY = decay_table is not None
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    V = v.shape[-1]
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty(B, T, HQ, V, device=q.device)

    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    L = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    M = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    w2 = torch.empty_like(w)
    G = HQ//H
    # q = torch.ones_like(q)
    # beta = torch.ones_like(beta)
    # k = torch.ones_like(k)
    # v = torch.ones_like(v)
    # w = torch.ones_like(w)
    # pdb.set_trace()
    intra_chunk_preprocess_fwd_kernel[grid](
        q=q, k=k, v=v, w=w, beta=beta, g_cumsum=g_cumsum,
        o=o, A=A, L=L, M=M, w2=w2, q_new=q_new, k_new=k_new,
        scale=scale, offsets=cu_seqlens, indices=indices, T=T,
        H=H, G=G, HQ=HQ, K=K, V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_G=(g_cumsum is not None),
        decay_table=decay_table,
        USE_DECAY=USE_DECAY,
        num_warps=4 if BT == 64 else 2,
    )
    # pdb.set_trace()
    return q_new, k_new, w2, o, L, M