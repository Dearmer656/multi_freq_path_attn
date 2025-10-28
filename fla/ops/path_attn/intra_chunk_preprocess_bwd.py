import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem


# episold
@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_bwd_kernel(
    q, k, w, w2, beta,
    AT,
    dA_local, dq, dq_orig, dtheta, dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T,
    offsets, indices, theta, dtheta_inter, wavelet_decay_table,
    HQ: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_WAVELET_DECAY: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_beta = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dT = tl.zeros([BT, BT], dtype=tl.float32)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_Twb = tl.load(p_w2, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)
    s_theta = tl.load(theta + i_h)

    o_i = tl.arange(0, BT)
    b_qw = tl.where(o_i[:, None] >= o_i[None, :], tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, tl.trans(b_k)), 0).to(b_k.dtype)
    b_Twbk = tl.dot(b_T, b_wbk).to(b_w.dtype)

    p_dA_local = tl.make_block_ptr(dA_local + (bos * HQ + i_hq) * BT, (T, BT), (BT*HQ, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA_local = tl.load(p_dA_local, boundary_check=(0, 1))

    # # Twb part qw part.
    p_dq_orig = tl.make_block_ptr(dq_orig + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq_orig = tl.load(p_dq_orig, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))

    p_dw1 = tl.make_block_ptr(dw1 + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dw += tl.load(p_dw1, boundary_check=(0, 1))

    b_dqw = -tl.dot(b_dA_local, tl.trans(b_Twbk)) - tl.dot(b_dq.to(b_Twb.dtype), tl.trans(b_Twb))
    p_dw2 = tl.make_block_ptr(dw2 + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dTwb = -tl.dot(tl.trans(b_qw), b_dq) + tl.load(p_dw2, boundary_check=(0, 1))
    b_dT += tl.dot(b_dTwb.to(b_w_beta.dtype), tl.trans(b_w_beta))
    b_dw_beta += tl.dot(tl.trans(b_T), b_dTwb.to(b_T.dtype))

    b_dqw = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dqw, 0)
    b_dq += tl.dot(b_dA_local.to(b_k.dtype), b_k)
    b_dq += tl.dot(b_dqw.to(b_w.dtype), b_w)
    b_dq += b_dq_orig
    ##############################
    # 2025/10/25 edit
    # wavelet decay table backward
    ##############################
    if USE_WAVELET_DECAY:
        b_dtheta = tl.zeros([], dtype=tl.float32)
        rows = tl.arange(0, BT)  # (BT,)
        for tt in tl.static_range(0, BT):
            # 取固定 query 行 i=tt，对应 keys 的本地块 j∈[0, BT)
            base = wavelet_decay_table + tt * BT
            p_wdec = tl.make_block_ptr(
                base, (BK, BT),           # (BK, BT)
                (BT * BT, 1),             # strides: 沿 K 为 BT*BT，沿 j 为 1
                (0, 0), (BK, BT),
                (1, 0)
            )
            b_decay_KBT = s_theta * tl.load(p_wdec, boundary_check=(0, 1)).to(tl.float32)  # (BK, BT)
            b_decay_BTK = tl.trans(b_decay_KBT)                                  # (BT, BK)

            # 整块贡献：(BT, BT) @ (BT, BK) -> (BT, BK)
            contrib_all = tl.dot(b_dA_local.to(tl.float32), b_decay_BTK)         # (BT, BK)

            # 仅向第 tt 行写入增量（用 float 掩码，避免 bool 参与算术）
            row_mask_f32 = (rows == tt).to(tl.float32)[:, None]                  # (BT, 1)
            b_dq += (row_mask_f32 * contrib_all).to(b_dq.dtype)          # (BT, BK)
            p_q_row = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1),
                (i_t * BT + tt, 0),
                (1, BK),
                (1, 0)
            )
            d_q_row_2d = tl.load(p_q_row, boundary_check=(0, 1)).to(tl.float32)  # (1, BK)
            row_k = tl.reshape(d_q_row_2d, (K,))   # (K,)
            q_wave_local = tl.sum(b_decay_KBT * row_k[:, None], axis=0)         # (BT,)
            # q_wave_local_2d = tl.dot(d_q_row_2d, b_decay_KBT)  # (BT,) ### 之所以错 是因为 我估计是因为生成的BTK覆盖了原本的KBT。
            b_dA_row = tl.sum(b_dA_local * row_mask_f32, axis=0)  # (BK,)
            dA_q_wave_local = tl.sum(b_dA_row * q_wave_local)
            b_dtheta += dA_q_wave_local
        b_dtheta = tl.load(dtheta_inter + i_h) + b_dtheta
        tl.atomic_add(dtheta + i_h, b_dtheta.to(dtheta.dtype.element_ty))
    ##############################
    b_dw += tl.dot(tl.trans(b_dqw.to(b_q.dtype)), b_q)
    p_q_new = tl.make_block_ptr(dq_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))

    # Twbk part
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))
    b_dTwbk = -tl.dot(tl.trans(b_qw), b_dA_local.to(b_qw.dtype)) - tl.dot(b_w, tl.trans(b_dk.to(b_w.dtype)))
    b_dw -= tl.dot(b_Twbk, b_dk.to(b_w.dtype))
    b_dT += tl.dot(b_dTwbk.to(b_wbk.dtype), tl.trans(b_wbk))
    b_dwbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(tl.trans(b_T), b_dTwbk.to(b_T.dtype)), 0).to(b_w.dtype)
    b_dw_beta += tl.dot(b_dwbk, b_k)

    b_dk += tl.dot(tl.trans(b_dwbk), b_w_beta)
    b_dk += tl.dot(tl.trans(b_dA_local), b_q)
    p_dk_new = tl.make_block_ptr(dk_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk_new, b_dk.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

    # matrix inverse's gradient
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (BT, T), (1, BT*H), (0, i_t * BT), (BT, BT), (0, 1))
    b_Tt = tl.load(p_T, boundary_check=(0, 1))
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0).to(b_w.dtype)
    b_dT = tl.dot(b_Tt, b_dT).to(b_w.dtype)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0).to(b_k.dtype)

    b_dw_beta += tl.dot(b_dT, b_w)
    b_dw += tl.dot(tl.trans(b_dT), b_w_beta)
    b_dw += b_dw_beta * b_beta[:, None]
    b_dbeta = tl.sum(b_dw_beta * b_w, axis=1)

    p_dw = tl.make_block_ptr(dw + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dw, b_dw.to(dw.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + (bos * HQ + i_hq), (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_dbeta, b_dbeta.to(dbeta.dtype.element_ty), boundary_check=(0, ))


def intra_chunk_preprocess_bwd_fn(q, k, w, w2, beta,
                                  dq, dq_orig, dk, dA_local,
                                  dw1, dw2,
                                  A, L, D, do, scale, cu_seqlens=None,
                                  theta=None, dtheta_inter=None,
                                  wavelet_decay_table=None, USE_WAVELET_DECAY=False):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    # better precision because h would be of norm smaller than 1 anyways

    dbeta = torch.empty(B, T, HQ, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dw = torch.empty(B, T, HQ, K, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dk_new = torch.empty_like(dk, dtype=k.dtype if G == 1 else torch.float32)  # float32 reduction
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    dtheta = torch.zeros_like(theta, dtype=torch.float32)
    intra_chunk_preprocess_bwd_kernel[grid](
        q=q, k=k, w=w, w2=w2, beta=beta,
        AT=A,
        dA_local=dA_local, dq=dq, dq_orig=dq_orig, dtheta=dtheta, dq_new=dq_new, dk=dk, dk_new=dk_new, dw=dw, dbeta=dbeta, dw1=dw1, dw2=dw2, T=T,
        offsets=cu_seqlens, indices=indices,
        HQ=HQ, G=G, H=H,
        K=K, BT=BT, BK=triton.next_power_of_2(K),
        num_stages=3 if check_shared_mem('hopper') else 1,
        theta=theta, dtheta_inter=dtheta_inter,
        wavelet_decay_table=wavelet_decay_table if USE_WAVELET_DECAY else None,
        USE_WAVELET_DECAY= 1 if USE_WAVELET_DECAY else 0,
    )
    return dq_new, dk_new, dbeta, dw, dtheta
