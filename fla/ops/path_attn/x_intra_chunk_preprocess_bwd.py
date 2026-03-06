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
    q, k, w, beta,
    AT,
    dA_local, dq, dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T,
    offsets, indices,
    HQ: tl.constexpr, G: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    decay_table,
    USE_DECAY: tl.constexpr,
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

    p_q    = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k    = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_w    = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_T    = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # loads
    b_w    = tl.load(p_w, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_q    = tl.load(p_q, boundary_check=(0, 1))
    b_k    = tl.load(p_k, boundary_check=(0, 1))
    b_T    = tl.load(p_T, boundary_check=(0, 1))
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]

    # forward intermediates reproduced
    b_qw  = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T).to(b_q.dtype)
    b_wbk_base = tl.dot(b_w, tl.trans(b_k)).to(b_q.dtype)

    # coeff
    if USE_DECAY:
        T_tile = tl.minimum(BT, T - i_t * BT)
        delta  = (o_i[:, None] - o_i[None, :]).to(tl.int32)
        delta  = tl.maximum(delta, 0)
        delta  = tl.minimum(delta, T_tile - 1)
        i_hr   = i_h * G + (i_hq % G)
        row_ptr = decay_table + i_hr * BT
        kappa   = tl.load(row_ptr + delta)
        coeff   = (b_beta[None, :] * kappa).to(b_q.dtype)
    else:
        coeff   = b_beta[None, :].to(b_q.dtype)

    b_wbk  = tl.where(o_i[:, None] > o_i[None, :], coeff * b_wbk_base, 0).to(b_q.dtype)
    b_Twbk = tl.dot(b_T, b_wbk).to(b_w.dtype)

    # grads input
    p_dA_local = tl.make_block_ptr(dA_local + (bos * HQ + i_hq) * BT, (T, BT), (BT*HQ, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA_local = tl.load(p_dA_local, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))

    # new: decay path grads
    b_weighted = (b_qwT * coeff).to(b_w.dtype)                      # [BT,BT]
    b_dw      += - tl.dot(tl.trans(b_weighted), b_dq.to(b_w.dtype)) # dW from q-update
    b_GW_T     = tl.dot(b_dq.to(b_w.dtype), tl.trans(b_w))          # [BT,BT]
    b_dQW      = (- b_GW_T * coeff).to(b_q.dtype)                   # d(QW) from q-update
    b_dT      += tl.dot(tl.trans(b_qw.to(b_dQW.dtype)), b_dQW).to(b_dT.dtype)

    # merge A-path and q-update path contributions to dq/dw
    b_dqw  = -tl.dot(b_dA_local, tl.trans(b_Twbk))
    b_dqw += b_dQW
    b_dqw  = tl.where(m_t, b_dqw, 0)

    b_dq += tl.dot(b_dA_local.to(b_k.dtype), b_k)
    b_dq += tl.dot(b_dqw.to(b_w.dtype), b_w)
    b_dw += tl.dot(tl.trans(b_dqw.to(b_q.dtype)), b_q)

    # write dq_new
    p_q_new = tl.make_block_ptr(dq_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1),
                                (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))

    # Twbk branch (unchanged)
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))
    b_dTwbk = -tl.dot(tl.trans(b_qw), b_dA_local.to(b_qw.dtype)) - tl.dot(b_w, tl.trans(b_dk.to(b_w.dtype)))
    b_dw   -= tl.dot(b_Twbk, b_dk.to(b_w.dtype))
    b_dT   += tl.dot(b_dTwbk.to(b_wbk.dtype), tl.trans(b_wbk))
    b_dwbk  = tl.where(o_i[:, None] > o_i[None, :],
                       tl.dot(tl.trans(b_T), b_dTwbk.to(b_T.dtype)), 0).to(b_w.dtype)

    b_dk   += tl.dot(tl.trans(b_dwbk), b_w_beta)
    b_dk   += tl.dot(tl.trans(b_dA_local), b_q)
    p_dk_new = tl.make_block_ptr(dk_new + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk_new, b_dk.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

    # dT through triangular solve
    p_Tt = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (BT, T), (1, BT*H), (0, i_t * BT), (BT, BT), (0, 1))
    b_Tt = tl.load(p_Tt, boundary_check=(0, 1))
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0).to(b_w.dtype)
    b_dT = tl.dot(b_Tt, b_dT).to(b_w.dtype)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0).to(b_k.dtype)

    # fold structural path beta & w grads
    b_dw_beta += tl.dot(b_dT, b_w)
    b_dw      += tl.dot(tl.trans(b_dT), b_w_beta)
    b_dw      += b_dw_beta * b_beta[:, None]
    b_dbeta    = tl.sum(b_dw_beta * b_w, axis=1)

    # stores
    p_dw    = tl.make_block_ptr(dw + (bos * HQ + i_hq) * K, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dw, b_dw.to(dw.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + (bos * HQ + i_hq), (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_dbeta, b_dbeta.to(dbeta.dtype.element_ty), boundary_check=(0, ))


def intra_chunk_preprocess_bwd_fn(q, k, w, beta,
                                  dq, dk, dA_local,
                                  dw1, dw2,
                                  A, L, D, do, scale, cu_seqlens=None, decay_table=None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    # better precision because h would be of norm smaller than 1 anyways
    USE_DECAY = decay_table is not None
    dbeta = torch.empty(B, T, HQ, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dw = torch.empty(B, T, HQ, K, device=q.device, dtype=k.dtype if G == 1 else torch.float32)
    dk_new = torch.empty_like(dk, dtype=k.dtype if G == 1 else torch.float32)  # float32 reduction
    dq_new = torch.empty_like(dq, dtype=q.dtype)

    intra_chunk_preprocess_bwd_kernel[grid](
        q=q, k=k, w=w, beta=beta,
        AT=A,
        dA_local=dA_local, dq=dq, dq_new=dq_new, dk=dk, dk_new=dk_new, dw=dw, dbeta=dbeta, dw1=dw1, dw2=dw2, T=T,
        offsets=cu_seqlens, indices=indices,
        HQ=HQ, G=G, H=H,
        K=K, BT=BT, BK=triton.next_power_of_2(K),
        num_stages=3 if check_shared_mem('hopper') else 1,
        decay_table=decay_table,
        USE_DECAY=USE_DECAY,
    )
    return dq_new, dk_new, dbeta, dw
