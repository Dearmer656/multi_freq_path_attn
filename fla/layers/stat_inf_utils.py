import os
import torch
import torch.distributed as dist

def _get_dist_ids():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
    return rank, local_rank, world_size, dev

@torch.no_grad()
def _quantiles_flat(x: torch.Tensor, qs=(0.1, 0.5, 0.9, 0.95, 0.99)):
    # flatten on CPU to reduce GPU overhead
    y = x.detach()
    if y.is_cuda:
        y = y.float().cpu()
    else:
        y = y.float()
    y = y.flatten()
    if y.numel() == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = torch.quantile(y, q).item()
    return out

@torch.no_grad()
def _router_gate_stats(
    *,
    router_name: str,
    logits: torch.Tensor,   # [B,T,H,S]
    gate: torch.Tensor,     # [B,T,H,S]
    tau: float,
    global_step: int,
    log_every: int,
    layer_idx: int,
    logger_obj=None,        # e.g. self.logger, or None->print
):
    if log_every <= 0:
        return
    if global_step < 0 or (global_step % log_every) != 0:
        return

    rank, local_rank, world_size, dev = _get_dist_ids()

    # -------- logits stats --------
    z = logits
    z_mean = z.float().mean().item()
    z_std  = z.float().std(unbiased=False).item()
    z_abs_p99 = _quantiles_flat(z.abs(), qs=(0.99,))["p99"]

    # norm over last dim S -> [B,T,H]
    z_norm = torch.linalg.vector_norm(z.float(), ord=2, dim=-1)
    z_norm_q = _quantiles_flat(z_norm, qs=(0.5, 0.9, 0.99))

    # -------- gate stats --------
    g = gate.float()
    eps = 1e-9
    ent = -(g * (g + eps).log()).sum(dim=-1)  # [B,T,H]
    top2, _ = torch.topk(g, k=2, dim=-1)
    top1 = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]

    S = g.shape[-1]
    idx = torch.arange(S, device=g.device, dtype=g.dtype)
    eshift = (g * idx).sum(dim=-1)  # [B,T,H]

    ent_mean = ent.mean().item()
    top1_mean = top1.mean().item()
    margin_mean = margin.mean().item()
    es_mean = eshift.mean().item()
    es_std = eshift.std(unbiased=False).item()

    ent_q = _quantiles_flat(ent, qs=(0.1, 0.5, 0.9))
    top1_q = _quantiles_flat(top1, qs=(0.5, 0.9, 0.99))
    margin_q = _quantiles_flat(margin, qs=(0.5, 0.9, 0.99))

    msg = (
        f"[router stats] rank={rank}/{world_size} local_rank={local_rank} cuda={dev} "
        f"layer={layer_idx} step={global_step} router={router_name} tau={tau:.4g} | "
        f"logits mean={z_mean:.4e} std={z_std:.4e} abs_p99={z_abs_p99:.4e} "
        f"norm_p50={z_norm_q['p50']:.4e} norm_p90={z_norm_q['p90']:.4e} norm_p99={z_norm_q['p99']:.4e} | "
        f"entropy mean={ent_mean:.4f} p10={ent_q['p10']:.4f} p50={ent_q['p50']:.4f} p90={ent_q['p90']:.4f} | "
        f"top1 mean={top1_mean:.4f} p50={top1_q['p50']:.4f} p90={top1_q['p90']:.4f} p99={top1_q['p99']:.4f} | "
        f"margin mean={margin_mean:.4f} p50={margin_q['p50']:.4f} p90={margin_q['p90']:.4f} p99={margin_q['p99']:.4f} | "
        f"Eshift mean={es_mean:.4f} std={es_std:.4f}"
    )

    if logger_obj is not None:
        # supports logger.info(...)
        try:
            logger_obj.info(msg)
        except Exception:
            print(msg)
    else:
        print(msg)
