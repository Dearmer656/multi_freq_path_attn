from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn


TensorOrScalar = Union[torch.Tensor, float, int]


@dataclass
class RouterNormConfig:
    enable: bool = False
    mode: str = "pre_gate"  # pre_gate | post_gate | logit | none
    norm_type: str = "rmsnorm"  # layernorm | rmsnorm | zscore | none
    affine: bool = False
    eps: float = 1e-5
    clamp_std_min: float = 1e-4
    log_every: int = 500
    log_heads: list[int] = field(default_factory=lambda: [0, 3, 7])
    log_tokens: list[Union[int, str]] = field(default_factory=lambda: [0, -1])

    def normalize(self) -> "RouterNormConfig":
        mode = str(self.mode).strip().lower()
        if mode not in {"pre_gate", "post_gate", "logit", "none"}:
            mode = "pre_gate"
        self.mode = mode

        norm_type = str(self.norm_type).strip().lower()
        if norm_type not in {"layernorm", "rmsnorm", "zscore", "none"}:
            norm_type = "rmsnorm"
        self.norm_type = norm_type

        self.enable = bool(self.enable)
        self.affine = bool(self.affine)
        self.eps = float(self.eps)
        self.clamp_std_min = float(self.clamp_std_min)
        self.log_every = max(0, int(self.log_every))
        self.log_heads = _parse_heads(self.log_heads)
        self.log_tokens = _parse_tokens(self.log_tokens)
        return self


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        x = v.strip().lower()
        if x in ("1", "true", "yes", "y", "on"):
            return True
        if x in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _split_csv(x: str) -> list[str]:
    return [p.strip() for p in str(x).split(",") if p.strip()]


def _parse_heads(v: Any) -> list[int]:
    if v is None:
        return [0, 3, 7]
    if isinstance(v, str):
        vals = _split_csv(v)
    elif isinstance(v, (list, tuple, set)):
        vals = list(v)
    else:
        vals = [v]
    out = []
    for item in vals:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out if out else [0, 3, 7]


def _parse_tokens(v: Any) -> list[Union[int, str]]:
    if v is None:
        return [0, -1]
    if isinstance(v, str):
        vals = _split_csv(v)
    elif isinstance(v, (list, tuple, set)):
        vals = list(v)
    else:
        vals = [v]
    out: list[Union[int, str]] = []
    for item in vals:
        if isinstance(item, str):
            t = item.strip().lower()
            if t in ("mid", "middle", "center"):
                out.append("mid")
                continue
            try:
                out.append(int(t))
            except Exception:
                continue
        else:
            try:
                out.append(int(item))
            except Exception:
                continue
    return out if out else [0, -1]


def build_router_norm_config(config: Any) -> RouterNormConfig:
    src = getattr(config, "router_norm", None)
    src = src if isinstance(src, dict) else {}
    cfg = RouterNormConfig(
        enable=_as_bool(src.get("enable", getattr(config, "router_norm_enable", False)), default=False),
        mode=str(src.get("mode", getattr(config, "router_norm_mode", "pre_gate"))),
        norm_type=str(src.get("type", src.get("norm_type", getattr(config, "router_norm_type", "rmsnorm")))),
        affine=_as_bool(src.get("affine", getattr(config, "router_norm_affine", False)), default=False),
        eps=_as_float(src.get("eps", getattr(config, "router_norm_eps", 1e-5)), 1e-5),
        clamp_std_min=_as_float(
            src.get("clamp_std_min", getattr(config, "router_norm_clamp_std_min", 1e-4)), 1e-4
        ),
        log_every=_as_int(src.get("log_every", getattr(config, "router_norm_log_every", 500)), 500),
        log_heads=src.get("log_heads", getattr(config, "router_norm_log_heads", [0, 3, 7])),
        log_tokens=src.get("log_tokens", getattr(config, "router_norm_log_tokens", [0, -1])),
    )
    return cfg.normalize()


class RouterNorm(nn.Module):
    """
    fp32-stat normalization over the last dimension.
    Supports: layernorm, rmsnorm, zscore, none.
    """

    def __init__(
        self,
        norm_type: str = "rmsnorm",
        eps: float = 1e-5,
        clamp_std_min: float = 1e-4,
        affine: bool = False,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        norm_type = str(norm_type).strip().lower()
        if norm_type not in {"layernorm", "rmsnorm", "zscore", "none"}:
            norm_type = "rmsnorm"
        self.norm_type = norm_type
        self.eps = float(eps)
        self.clamp_std_min = float(clamp_std_min)
        self.affine = bool(affine)
        self.feature_dim = int(feature_dim) if feature_dim is not None else None

        self.weight: Optional[nn.Parameter]
        self.bias: Optional[nn.Parameter]
        if self.affine and self.feature_dim is not None and self.feature_dim > 0:
            self.weight = nn.Parameter(torch.ones(self.feature_dim, dtype=torch.float32))
            if self.norm_type in {"layernorm", "zscore"}:
                self.bias = nn.Parameter(torch.zeros(self.feature_dim, dtype=torch.float32))
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    @property
    def enabled(self) -> bool:
        return self.norm_type != "none"

    def _can_apply_affine(self, feature_dim: int, needs_bias: bool) -> bool:
        if not self.affine:
            return False
        if self.weight is None or self.weight.numel() != int(feature_dim):
            return False
        if needs_bias and (self.bias is None or self.bias.numel() != int(feature_dim)):
            return False
        return True

    def _apply_affine(self, y: torch.Tensor, needs_bias: bool) -> torch.Tensor:
        feature_dim = int(y.shape[-1])
        if not self._can_apply_affine(feature_dim, needs_bias):
            return y
        shape = [1] * (y.dim() - 1) + [feature_dim]
        w = self.weight.to(device=y.device, dtype=y.dtype).view(shape)
        y = y * w
        if needs_bias:
            b = self.bias.to(device=y.device, dtype=y.dtype).view(shape)
            y = y + b
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "none":
            return x
        in_dtype = x.dtype
        # Guard against non-finite upstream values (e.g., from q_corr/M solve path).
        # Keep this local to router-norm branch so training can proceed while logs surface issues.
        xf = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.norm_type in {"layernorm", "zscore"}:
            mean = xf.mean(dim=-1, keepdim=True)
            var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
            std = torch.sqrt(var + self.eps).clamp_min(self.clamp_std_min)
            y = (xf - mean) / std
            y = self._apply_affine(y, needs_bias=True)
        elif self.norm_type == "rmsnorm":
            rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps).clamp_min(self.clamp_std_min)
            y = xf / rms
            y = self._apply_affine(y, needs_bias=False)
        else:
            y = xf
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y.to(dtype=in_dtype)


class RouterNormStatsLogger:
    def __init__(
        self,
        log_every: int = 500,
        log_heads: Optional[Sequence[int]] = None,
        log_tokens: Optional[Sequence[Union[int, str]]] = None,
        eps: float = 1e-12,
    ):
        self.log_every = max(0, int(log_every))
        self.log_heads = list(log_heads) if log_heads is not None else [0, 3, 7]
        self.log_tokens = list(log_tokens) if log_tokens is not None else [0, -1]
        self.eps = float(eps)

    @staticmethod
    def _safe_step(step: Any) -> Optional[int]:
        if step is None:
            return None
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                return None
            step = step.detach().item()
        try:
            return int(step)
        except Exception:
            return None

    def should_log(self, step: Any) -> bool:
        s = self._safe_step(step)
        if s is None:
            return False
        if self.log_every <= 0:
            return False
        return s >= 0 and (s % self.log_every == 0)

    @staticmethod
    def _resolve_indices(spec: Sequence[Union[int, str]], size: int) -> list[int]:
        out: list[int] = []
        for item in spec:
            if isinstance(item, str):
                t = item.strip().lower()
                if t in ("mid", "middle", "center"):
                    idx = size // 2
                else:
                    try:
                        idx = int(t)
                    except Exception:
                        continue
            else:
                try:
                    idx = int(item)
                except Exception:
                    continue
            if idx < 0:
                idx += size
            if 0 <= idx < size:
                out.append(idx)
        dedup = []
        seen = set()
        for i in out:
            if i not in seen:
                dedup.append(i)
                seen.add(i)
        return dedup

    @torch.no_grad()
    def log_header(self, step: Any, layer_idx: Optional[int], mode: str, norm_type: str, eps: float):
        s = self._safe_step(step)
        if s is None:
            return
        layer_repr = "NA" if layer_idx is None else str(int(layer_idx))
        print(
            f"[RouterNorm] step={s} layer={layer_repr} mode={mode} type={norm_type} eps={float(eps):.6g}"
        )

    @torch.no_grad()
    def _log_vec_stats(
        self,
        tag: str,
        vec: torch.Tensor,
        b: int,
        t: int,
        h: int,
        add_entropy: bool = False,
    ):
        v = vec.detach().float()
        finite = torch.isfinite(v)
        numel = int(v.numel())
        n_finite = int(finite.sum().item())
        n_nan = int(torch.isnan(v).sum().item())
        n_inf = int(torch.isinf(v).sum().item())
        if n_finite > 0:
            vf = v[finite]
            mean = float(vf.mean().item())
            std = float(vf.std(unbiased=False).item())
            abs_mean = float(vf.abs().mean().item())
            vmin = float(vf.min().item())
            vmax = float(vf.max().item())
        else:
            mean = float("nan")
            std = float("nan")
            abs_mean = float("nan")
            vmin = float("nan")
            vmax = float("nan")
        msg = (
            f"[RouterNorm] {tag} (b={b},t={t},h={h}): "
            f"mean={mean:.6e} std={std:.6e} abs_mean={abs_mean:.6e} min={vmin:.6e} max={vmax:.6e} "
            f"finite={n_finite}/{numel} nan={n_nan} inf={n_inf}"
        )
        if add_entropy:
            v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            p = v_safe / v_safe.sum().clamp_min(self.eps)
            entropy = float(-(p * p.log()).sum().item())
            sparsity = float((v_safe.abs() < 1e-3).float().mean().item())
            msg += f" entropy={entropy:.6e} sparsity={sparsity:.6e}"
        print(msg)

    @torch.no_grad()
    def log_tensor_samples(self, tag: str, x: torch.Tensor, layout: str = "bths"):
        if x is None:
            return
        t = x.detach()
        if layout == "bhtn":
            if t.dim() != 4:
                return
            t = t.permute(0, 2, 1, 3).contiguous()  # -> [B,T,H,N]
        if t.dim() == 3:
            t = t.unsqueeze(2)
        if t.dim() == 5:
            t = t.mean(dim=-1)
        if t.dim() != 4:
            return
        B, T, H, _ = t.shape
        if B == 0 or T == 0 or H == 0:
            return
        b = 0
        heads = self._resolve_indices(self.log_heads, H)
        tokens = self._resolve_indices(self.log_tokens, T)
        if not heads:
            heads = [0]
        if not tokens:
            tokens = [0]
        for ti in tokens:
            for hi in heads:
                self._log_vec_stats(tag=tag, vec=t[b, ti, hi], b=b, t=ti, h=hi)

    @torch.no_grad()
    def log_gate_samples(self, tag: str, g: torch.Tensor):
        if g is None:
            return
        gt = g.detach()
        if gt.dim() == 3:
            gt = gt.unsqueeze(2)
        if gt.dim() != 4:
            return
        B, T, H, _ = gt.shape
        if B == 0 or T == 0 or H == 0:
            return
        b = 0
        heads = self._resolve_indices(self.log_heads, H)
        tokens = self._resolve_indices(self.log_tokens, T)
        if not heads:
            heads = [0]
        if not tokens:
            tokens = [0]
        for ti in tokens:
            for hi in heads:
                self._log_vec_stats(tag=tag, vec=gt[b, ti, hi], b=b, t=ti, h=hi, add_entropy=True)


def ensure_q_bths(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 4:
        return x, False
    if x.dim() == 3:
        return x.unsqueeze(2), True
    raise ValueError(f"Expected q shape [B,T,H,S] or [B,T,S], got {tuple(x.shape)}")


def _rms_over_last_dim(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    xf = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    rms = torch.sqrt(xf.pow(2).mean(dim=-1) + float(eps))
    return rms.to(dtype=x.dtype)


def prepare_gate(gate: Optional[TensorOrScalar], q_bths: torch.Tensor) -> torch.Tensor:
    B, T, H, S = q_bths.shape
    if gate is None:
        return torch.ones((1, 1, 1, 1), device=q_bths.device, dtype=q_bths.dtype)
    if isinstance(gate, (float, int)):
        return torch.tensor(float(gate), device=q_bths.device, dtype=q_bths.dtype).view(1, 1, 1, 1)
    g = gate.to(device=q_bths.device, dtype=q_bths.dtype)
    if g.dim() == 3:
        g = g.unsqueeze(2)
    if g.dim() != 4:
        raise ValueError(f"Expected gate shape [B,T,H,S] (broadcastable), got {tuple(g.shape)}")
    target = (B, T, H, S)
    for idx, (gs, ts) in enumerate(zip(g.shape, target)):
        if gs not in (1, ts):
            raise ValueError(
                f"Gate shape {tuple(g.shape)} is not broadcastable to {target} at dim={idx}"
            )
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def apply_router_norm_mode(
    q_like: torch.Tensor,
    gate: Optional[TensorOrScalar],
    p_s: torch.Tensor,
    router_norm: Optional[RouterNorm],
    cfg: RouterNormConfig,
    *,
    shift_sep_use: bool = False,
    logger: Optional[RouterNormStatsLogger] = None,
    step: Optional[int] = None,
    layer_idx: Optional[int] = None,
    tensor_name: str = "q_s",
    gate_name: str = "gate",
    emit_header: bool = False,
) -> torch.Tensor:
    """
    Returns rel with shape [B,H,T,N].
    - shift_sep_use=False:
        q_like: [B,T,H,S] or [B,T,S], p_s: [S,T,N]
    - shift_sep_use=True:
        q_like: [B,T,H,S,C], p_s: [S,C,T,N]
    """
    cfg = cfg.normalize()
    mode = cfg.mode
    norm_is_active = bool(cfg.enable and (mode != "none") and router_norm is not None and router_norm.enabled)
    do_log = bool(cfg.enable and logger is not None and logger.should_log(step))
    p_s = torch.nan_to_num(p_s, nan=0.0, posinf=0.0, neginf=0.0)
    if do_log and emit_header:
        logger.log_header(step=step, layer_idx=layer_idx, mode=mode, norm_type=cfg.norm_type, eps=cfg.eps)

    if not shift_sep_use:
        q_bths, _ = ensure_q_bths(q_like)
        g = prepare_gate(gate, q_bths)

        if do_log:
            logger.log_gate_samples(f"{gate_name}", g)

        if mode == "pre_gate":
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} pre", q_bths, layout="bths")
            q_post = router_norm(q_bths) if norm_is_active else q_bths
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} post", q_post, layout="bths")
            rel = torch.einsum("b t h s, s t n -> b h t n", g * q_post, p_s)
        elif mode == "post_gate":
            u_pre = g * q_bths
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} pre", u_pre, layout="bths")
            u_post = router_norm(u_pre) if norm_is_active else u_pre
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} post", u_post, layout="bths")
            rel = torch.einsum("b t h s, s t n -> b h t n", u_post, p_s)
        elif mode == "logit":
            rel_pre = torch.einsum("b t h s, s t n -> b h t n", g * q_bths, p_s)
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} logit pre", rel_pre, layout="bhtn")
            rel = router_norm(rel_pre) if norm_is_active else rel_pre
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} logit post", rel, layout="bhtn")
        else:
            rel = torch.einsum("b t h s, s t n -> b h t n", g * q_bths, p_s)
            if do_log:
                logger.log_tensor_samples(f"{tensor_name} raw", q_bths, layout="bths")
        return rel

    if q_like.dim() != 5:
        raise ValueError(f"Expected q_like [B,T,H,S,C] when shift_sep_use=True, got {tuple(q_like.shape)}")
    B, T, H, S, C = q_like.shape
    q_bths = q_like.mean(dim=-1)
    g = prepare_gate(gate, q_bths)
    g_bthsc = g.unsqueeze(-1)

    if do_log:
        logger.log_gate_samples(f"{gate_name}", g)

    if mode == "pre_gate":
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} pre", _rms_over_last_dim(q_like), layout="bths")
        q_perm = q_like.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,H,C,S]
        q_post_perm = router_norm(q_perm) if norm_is_active else q_perm
        q_post = q_post_perm.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,H,S,C]
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} post", _rms_over_last_dim(q_post), layout="bths")
        rel = torch.einsum("b t h s c, s c t n -> b h t n", g_bthsc * q_post, p_s)
    elif mode == "post_gate":
        u_pre = g_bthsc * q_like
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} pre", _rms_over_last_dim(u_pre), layout="bths")
        u_perm = u_pre.permute(0, 1, 2, 4, 3).contiguous()
        u_post_perm = router_norm(u_perm) if norm_is_active else u_perm
        u_post = u_post_perm.permute(0, 1, 2, 4, 3).contiguous()
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} post", _rms_over_last_dim(u_post), layout="bths")
        rel = torch.einsum("b t h s c, s c t n -> b h t n", u_post, p_s)
    elif mode == "logit":
        rel_pre = torch.einsum("b t h s c, s c t n -> b h t n", g_bthsc * q_like, p_s)
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} logit pre", rel_pre, layout="bhtn")
        rel = router_norm(rel_pre) if norm_is_active else rel_pre
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} logit post", rel, layout="bhtn")
    else:
        rel = torch.einsum("b t h s c, s c t n -> b h t n", g_bthsc * q_like, p_s)
        if do_log:
            logger.log_tensor_samples(f"{tensor_name} raw", q_bths, layout="bths")
    return rel
