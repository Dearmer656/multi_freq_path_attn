import torch
import math

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

def geom_p_schedule(step: int, total_steps: int,
                    p_start: float = 0.25,   # 前期更偏近距（尾短）
                    p_end: float   = 0.05    # 后期更偏长距（尾长）
                   ) -> float:
    """
    返回当前 step 的几何分布参数 geom_p（用于 Δ ~ Geom(p) 截断到 [min_delta, max_delta]）。
    采用余弦退火从 p_start -> p_end 的平滑过渡。
    """
    if total_steps <= 0:
        return p_end
    t = max(0.0, min(1.0, step / total_steps))           # 0~1
    s = 0.5 * (1 - math.cos(math.pi * t))                # 余弦调度
    return float(p_start + (p_end - p_start) * s)