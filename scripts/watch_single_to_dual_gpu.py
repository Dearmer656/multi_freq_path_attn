#!/usr/bin/env python3
"""Watch a running 1-GPU fallback job and cancel it when 2-GPU placement is possible.

Typical use:

  python scripts/watch_single_to_dual_gpu.py \
    --fallback-job 123456 \
    --preferred-job 123457 \
    --interval 60 \
    --execute

The script is intentionally conservative:
  - It only cancels the explicit fallback job id.
  - By default it requires a preferred pending job to exist.
  - It requires the fallback job's node to have at least one additional free GPU
    of the same type, so the preferred 2-GPU job can run on that node.
  - Without --execute it is dry-run only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import time
from dataclasses import dataclass


GPU_ALIASES = {
    "6000": "6000",
    "rtx6000": "6000",
    "a6000": "a6000",
    "a100": "a100",
    "3090": "3090",
    "p6000": "p6000",
}


@dataclass
class Job:
    job_id: str
    state: str
    name: str
    node: str | None
    req_nodes: str | None
    exc_nodes: str | None
    gpu_type: str | None
    gpu_count: int
    command: str | None


@dataclass
class Node:
    name: str
    state: str
    gpu_type: str
    gpu_total: int


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def run_cmd(args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def normalize_gpu_type(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip().lower()
    return GPU_ALIASES.get(raw, raw)


def parse_gpu_from_tres(text: str | None) -> tuple[str | None, int]:
    if not text:
        return None, 0
    match = re.search(r"gres:gpu:([^:,\s]+):(\d+)", text)
    if match:
        return normalize_gpu_type(match.group(1)), int(match.group(2))
    match = re.search(r"gres/gpu=(\d+)", text)
    if match:
        return None, int(match.group(1))
    return None, 0


def parse_gpu_from_gres(text: str | None) -> tuple[str | None, int]:
    if not text:
        return None, 0
    match = re.search(r"gpu:([^:,\s]+):(\d+)", text)
    if match:
        return normalize_gpu_type(match.group(1)), int(match.group(2))
    return None, 0


def parse_kv_blob(blob: str) -> dict[str, str]:
    # scontrol prints fields like "JobId=... JobName=..." with embedded newlines.
    pairs = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", blob):
        pairs[key] = value
    return pairs


def get_job(job_id: str) -> Job | None:
    out = run_cmd(["scontrol", "show", "job", str(job_id)], check=False)
    if not out.strip() or "Invalid job id" in out or "slurm_load_jobs error" in out:
        return None
    kv = parse_kv_blob(out)
    state = kv.get("JobState", "UNKNOWN")
    node = kv.get("NodeList")
    if node in {None, "(null)", "None"}:
        node = None
    req_nodes = kv.get("ReqNodeList")
    if req_nodes in {None, "(null)", "None"}:
        req_nodes = None
    exc_nodes = kv.get("ExcNodeList")
    if exc_nodes in {None, "(null)", "None"}:
        exc_nodes = None
    tres_per_node = kv.get("TresPerNode")
    tres = kv.get("TRES")
    gpu_type, gpu_count = parse_gpu_from_tres(tres_per_node)
    if gpu_count == 0:
        gpu_type, gpu_count = parse_gpu_from_tres(tres)
    return Job(
        job_id=str(job_id),
        state=state,
        name=kv.get("JobName", ""),
        node=node,
        req_nodes=req_nodes,
        exc_nodes=exc_nodes,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        command=kv.get("Command"),
    )


def get_nodes() -> dict[str, Node]:
    out = run_cmd(["sinfo", "-N", "-h", "-o", "%N|%t|%G"])
    nodes: dict[str, Node] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        name, state, gres = parts
        gpu_type, gpu_total = parse_gpu_from_gres(gres)
        if gpu_type is None or gpu_total <= 0:
            continue
        nodes.setdefault(name, Node(name=name, state=state, gpu_type=gpu_type, gpu_total=gpu_total))
    return nodes


def get_running_gpu_usage() -> dict[str, int]:
    out = run_cmd(["squeue", "-h", "-t", "R", "-o", "%N|%b"])
    usage: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 2:
            continue
        node, tres = parts
        _, gpu_count = parse_gpu_from_tres(tres)
        if node and gpu_count > 0:
            usage[node] = usage.get(node, 0) + gpu_count
    return usage


def node_expr_allows(node: str, req_nodes: str | None, exc_nodes: str | None) -> bool:
    # Conservative parser: exact node name, empty, or simple comma/range expressions
    # containing the node are accepted. Unknown complex expressions are rejected.
    if exc_nodes:
        excluded = expand_simple_node_expr(exc_nodes)
        if node in excluded:
            return False
    if not req_nodes:
        return True
    allowed = expand_simple_node_expr(req_nodes)
    return node in allowed


def expand_simple_node_expr(expr: str) -> set[str]:
    # Supports: elm82, elm[71-73], elm71,elm73. Complex Slurm expressions are
    # intentionally treated as opaque to avoid false positives.
    result: set[str] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"([A-Za-z_-]+)\[(\d+)-(\d+)\]", part)
        if match:
            prefix, start_s, end_s = match.groups()
            width = len(start_s)
            for value in range(int(start_s), int(end_s) + 1):
                result.add(f"{prefix}{value:0{width}d}")
            continue
        if re.fullmatch(r"[A-Za-z_-]+\d+", part):
            result.add(part)
    return result


def check_once(args: argparse.Namespace) -> bool:
    preferred = None
    if args.preferred_job:
        preferred = get_job(args.preferred_job)
        if preferred is None:
            log(f"preferred job {args.preferred_job} no longer exists; exiting")
            return True
        if preferred.state != "PENDING":
            log(
                f"preferred job {preferred.job_id} state={preferred.state}; "
                "preferred is no longer pending, exiting"
            )
            return True

    fallback = get_job(args.fallback_job)
    if fallback is None:
        log(f"fallback job {args.fallback_job} no longer exists; exiting")
        return True
    if fallback.state != "RUNNING":
        log(f"fallback job {fallback.job_id} state={fallback.state}; waiting")
        return False
    if fallback.node is None:
        log(f"fallback job {fallback.job_id} has no assigned node yet; waiting")
        return False
    if fallback.gpu_count != 1:
        log(
            f"fallback job {fallback.job_id} uses {fallback.gpu_count} GPU(s), "
            "expected exactly 1; refusing to cancel"
        )
        return False

    if args.preferred_job:
        if preferred.gpu_count < args.preferred_gpus:
            log(
                f"preferred job {preferred.job_id} asks for {preferred.gpu_count} GPU(s), "
                f"expected >= {args.preferred_gpus}; waiting"
            )
            return False
    elif not args.force_without_preferred:
        raise RuntimeError("--preferred-job is required unless --force-without-preferred is set")

    nodes = get_nodes()
    node = nodes.get(fallback.node)
    if node is None:
        log(f"node {fallback.node} not found in sinfo GPU nodes; waiting")
        return False
    if node.gpu_type != fallback.gpu_type and fallback.gpu_type is not None:
        log(
            f"node GPU type {node.gpu_type} does not match fallback request "
            f"{fallback.gpu_type}; waiting"
        )
        return False

    usage = get_running_gpu_usage()
    used = usage.get(node.name, 0)
    free = max(0, node.gpu_total - used)

    if preferred is not None:
        if preferred.gpu_type is not None and preferred.gpu_type != node.gpu_type:
            log(
                f"preferred GPU type {preferred.gpu_type} does not match node "
                f"{node.name}:{node.gpu_type}; waiting"
            )
            return False
        if not node_expr_allows(node.name, preferred.req_nodes, preferred.exc_nodes):
            log(
                f"preferred job constraints do not allow node {node.name} "
                f"(ReqNodeList={preferred.req_nodes}, ExcNodeList={preferred.exc_nodes}); waiting"
            )
            return False

    # The fallback currently occupies one GPU. If there is at least one additional
    # free GPU, cancelling fallback should expose enough GPUs for the 2-GPU job.
    if free < args.min_free:
        log(
            f"fallback={fallback.job_id} node={node.name} used={used}/{node.gpu_total} "
            f"free={free}; need free>={args.min_free}; waiting"
        )
        return False

    action = (
        f"cancel fallback job {fallback.job_id} on {node.name}: "
        f"used={used}/{node.gpu_total}, free={free}, gpu_type={node.gpu_type}"
    )
    if args.execute:
        log(f"EXECUTE: {action}")
        run_cmd(["scancel", str(fallback.job_id)])
    else:
        log(f"DRY-RUN: would {action}. Add --execute to actually scancel.")
    return bool(args.exit_after_cancel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancel a running 1-GPU fallback job when its node has another free GPU."
    )
    parser.add_argument("--fallback-job", required=True, help="Explicit 1-GPU job id to cancel.")
    parser.add_argument("--preferred-job", help="Pending preferred 2-GPU job id.")
    parser.add_argument("--preferred-gpus", type=int, default=2)
    parser.add_argument("--min-free", type=int, default=1, help="Additional free GPUs needed on fallback node.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument("--execute", action="store_true", help="Actually run scancel. Default is dry-run.")
    parser.add_argument(
        "--force-without-preferred",
        action="store_true",
        help="Allow cancellation without checking a pending preferred job.",
    )
    parser.add_argument(
        "--exit-after-cancel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit after cancellation condition is met.",
    )
    parser.add_argument("--once", action="store_true", help="Run one check and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.min_free <= 0:
        raise SystemExit("--min-free must be positive")
    log(
        "start watch "
        f"fallback={args.fallback_job} preferred={args.preferred_job or 'none'} "
        f"interval={args.interval}s execute={int(args.execute)}"
    )
    while True:
        try:
            should_exit = check_once(args)
        except Exception as exc:
            log(f"ERROR: {exc}")
            should_exit = False
        if args.once or should_exit:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
