# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.router_norm import (
    RouterNorm,
    RouterNormConfig,
    RouterNormStatsLogger,
    apply_router_norm_mode,
)


def _cfg(mode="pre_gate", norm_type="rmsnorm", enable=True):
    return RouterNormConfig(
        enable=enable,
        mode=mode,
        norm_type=norm_type,
        affine=False,
        eps=1e-5,
        clamp_std_min=1e-4,
        log_every=500,
        log_heads=[0, 3, 7],
        log_tokens=[0, "mid", -1],
    ).normalize()


def test_router_norm_shape_invariance_q_shapes():
    torch.manual_seed(0)
    B, T, H, S, N = 2, 9, 4, 8, 7
    p_s = torch.randn(S, T, N)

    cfg = _cfg(mode="pre_gate", norm_type="rmsnorm", enable=True)
    norm = RouterNorm(norm_type=cfg.norm_type, eps=cfg.eps, clamp_std_min=cfg.clamp_std_min)

    q4 = torch.randn(B, T, H, S)
    gate4 = torch.randn(B, 1, H, S)
    rel4 = apply_router_norm_mode(
        q_like=q4,
        gate=gate4,
        p_s=p_s,
        router_norm=norm,
        cfg=cfg,
    )
    assert rel4.shape == (B, H, T, N)

    q3 = torch.randn(B, T, S)
    rel3 = apply_router_norm_mode(
        q_like=q3,
        gate=1.0,
        p_s=p_s,
        router_norm=norm,
        cfg=cfg,
    )
    assert rel3.shape == (B, 1, T, N)


@pytest.mark.parametrize("gate_shape", [(1, 1, 4, 8), (2, 1, 4, 8), (1, 9, 4, 8), (2, 9, 4, 8)])
def test_router_norm_gate_broadcast(gate_shape):
    torch.manual_seed(1)
    B, T, H, S, N = 2, 9, 4, 8, 6
    q = torch.randn(B, T, H, S)
    p_s = torch.randn(S, T, N)
    gate = torch.randn(*gate_shape)

    cfg = _cfg(mode="pre_gate", norm_type="none", enable=True)
    norm = RouterNorm(norm_type=cfg.norm_type, eps=cfg.eps, clamp_std_min=cfg.clamp_std_min)
    rel = apply_router_norm_mode(q_like=q, gate=gate, p_s=p_s, router_norm=norm, cfg=cfg)

    gate_ref = gate.to(dtype=q.dtype).expand(B, T, H, S)
    rel_ref = torch.einsum("b t h s, s t n -> b h t n", gate_ref * q, p_s)
    assert rel.shape == rel_ref.shape
    assert torch.allclose(rel, rel_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["pre_gate", "post_gate", "logit"])
@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm", "zscore"])
def test_router_norm_numerical_stability(mode, norm_type):
    torch.manual_seed(2)
    B, T, H, S, N = 2, 11, 3, 8, 9
    p_s = torch.randn(S, T, N)
    gate = torch.softmax(torch.randn(B, T, H, S), dim=-1)
    cfg = _cfg(mode=mode, norm_type=norm_type, enable=True)
    norm = RouterNorm(norm_type=cfg.norm_type, eps=cfg.eps, clamp_std_min=cfg.clamp_std_min)

    q_cases = [
        torch.zeros(B, T, H, S),
        torch.full((B, T, H, S), 1e-8) + 1e-10 * torch.randn(B, T, H, S),
        torch.randn(B, T, H, S) * 1e4,
    ]
    for q in q_cases:
        rel = apply_router_norm_mode(q_like=q, gate=gate, p_s=p_s, router_norm=norm, cfg=cfg)
        assert torch.isfinite(rel).all()


@pytest.mark.parametrize("mode", ["pre_gate", "post_gate", "logit"])
@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm", "zscore"])
def test_router_norm_non_finite_inputs_are_sanitized(mode, norm_type):
    torch.manual_seed(5)
    B, T, H, S, N = 2, 10, 3, 8, 6
    p_s = torch.randn(S, T, N)
    gate = torch.softmax(torch.randn(B, T, H, S), dim=-1)
    cfg = _cfg(mode=mode, norm_type=norm_type, enable=True)
    norm = RouterNorm(norm_type=cfg.norm_type, eps=cfg.eps, clamp_std_min=cfg.clamp_std_min)

    q = torch.randn(B, T, H, S)
    q[0, 0, 0, 0] = float("nan")
    q[0, 1, 1, 1] = float("inf")
    q[0, 2, 2, 2] = float("-inf")

    rel = apply_router_norm_mode(q_like=q, gate=gate, p_s=p_s, router_norm=norm, cfg=cfg)
    assert torch.isfinite(rel).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_norm_mixed_precision_fp32_reduction(dtype):
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == torch.float16 and device.type == "cpu":
        pytest.skip("float16 reduction test is unstable on CPU")

    B, T, H, S, N = 2, 8, 3, 16, 7
    q = (torch.randn(B, T, H, S, device=device, dtype=dtype) * 5.0).contiguous()
    gate = torch.softmax(torch.randn(B, T, H, S, device=device, dtype=torch.float32), dim=-1).to(dtype)
    p_s = torch.randn(S, T, N, device=device, dtype=dtype)

    cfg = _cfg(mode="pre_gate", norm_type="zscore", enable=True)
    norm = RouterNorm(
        norm_type=cfg.norm_type,
        eps=cfg.eps,
        clamp_std_min=cfg.clamp_std_min,
        affine=False,
    ).to(device)

    q_post = norm(q)
    assert q_post.dtype == dtype

    qf = q.float()
    mean = qf.mean(dim=-1, keepdim=True)
    var = (qf - mean).pow(2).mean(dim=-1, keepdim=True)
    std = torch.sqrt(var + cfg.eps).clamp_min(cfg.clamp_std_min)
    manual = ((qf - mean) / std).to(dtype=dtype)
    assert torch.allclose(q_post.float(), manual.float(), atol=3e-3, rtol=3e-3)

    rel = apply_router_norm_mode(
        q_like=q,
        gate=gate,
        p_s=p_s,
        router_norm=norm,
        cfg=cfg,
    )
    assert rel.dtype == dtype
    assert torch.isfinite(rel).all()


def test_router_norm_logger_output(capsys):
    torch.manual_seed(4)
    B, T, H, S, N = 1, 7, 4, 8, 6
    q = torch.randn(B, T, H, S)
    gate = torch.softmax(torch.randn(B, T, H, S), dim=-1)
    p_s = torch.randn(S, T, N)

    cfg = RouterNormConfig(
        enable=True,
        mode="pre_gate",
        norm_type="rmsnorm",
        affine=False,
        eps=1e-5,
        clamp_std_min=1e-4,
        log_every=1,
        log_heads=[0, 3],
        log_tokens=[0, "mid", -1],
    ).normalize()
    norm = RouterNorm(norm_type=cfg.norm_type, eps=cfg.eps, clamp_std_min=cfg.clamp_std_min)
    logger = RouterNormStatsLogger(log_every=1, log_heads=cfg.log_heads, log_tokens=cfg.log_tokens)

    _ = apply_router_norm_mode(
        q_like=q,
        gate=gate,
        p_s=p_s,
        router_norm=norm,
        cfg=cfg,
        logger=logger,
        step=123500,
        layer_idx=5,
        tensor_name="q_s",
        gate_name="gate1",
        emit_header=True,
    )
    out = capsys.readouterr().out
    assert "[RouterNorm] step=123500 layer=5 mode=pre_gate type=rmsnorm" in out
    assert "[RouterNorm] q_s pre" in out
    assert "[RouterNorm] q_s post" in out
    assert "[RouterNorm] gate1" in out
