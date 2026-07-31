#!/usr/bin/env python3
"""Recommend Slurm GPU placement from current queue state.

The script implements the local scheduling heuristic used for PaTH/QWAB runs:
prefer immediately available GPUs; otherwise prefer nodes where enough GPUs are
likely to free soon based on current job runtime and time limit.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import subprocess
import sys
from collections import defaultdict
from typing import Iterable


GPU_ALIASES = {
    "6000": "6000",
    "rtx6000": "6000",
    "a6000": "a6000",
    "a100": "a100",
    "3090": "3090",
}


@dataclasses.dataclass
class NodeInfo:
    name: str
    state: str
    gpu_type: str
    gpu_total: int
    cpu_alloc: int
    cpu_idle: int
    cpu_other: int
    cpu_total: int


@dataclasses.dataclass
class JobInfo:
    job_id: str
    partition: str
    name: str
    state: str
    runtime_s: int | None
    timelimit_s: int | None
    node: str
    gpu_type: str | None
    gpu_count: int

    @property
    def remaining_s(self) -> int | None:
        if self.runtime_s is None or self.timelimit_s is None:
            return None
        return max(0, self.timelimit_s - self.runtime_s)


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"failed to run {' '.join(args)}: {exc}") from exc


def parse_duration_s(text: str) -> int | None:
    text = text.strip()
    if not text or text in {"N/A", "INVALID", "UNLIMITED"}:
        return None
    if "-" in text:
        day_s, rest = text.split("-", 1)
        days = int(day_s)
    else:
        days = 0
        rest = text
    parts = rest.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        elif len(parts) == 1:
            h = 0
            m = 0
            s = int(parts[0])
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_gpu_type(raw: str) -> str:
    raw = raw.lower().strip()
    return GPU_ALIASES.get(raw, raw)


def parse_gres(gres: str) -> tuple[str, int] | None:
    # Examples: gpu:6000:8(S:0-1), gpu:a100:4, gpu:3090:8
    match = re.search(r"gpu:([^:,\s]+):(\d+)", gres)
    if not match:
        return None
    return normalize_gpu_type(match.group(1)), int(match.group(2))


def parse_tres_gpu(tres: str) -> tuple[str | None, int]:
    # Examples: gres:gpu:6000:4, gres/gpu=2, N/A
    match = re.search(r"gres:gpu:([^:,\s]+):(\d+)", tres)
    if match:
        return normalize_gpu_type(match.group(1)), int(match.group(2))
    match = re.search(r"gres/gpu=(\d+)", tres)
    if match:
        return None, int(match.group(1))
    return None, 0


def get_nodes() -> dict[str, NodeInfo]:
    out = run_cmd(["sinfo", "-N", "-h", "-o", "%N|%t|%G|%C"])
    nodes: dict[str, NodeInfo] = {}
    for line in out.splitlines():
        fields = line.split("|")
        if len(fields) != 4:
            continue
        name, state, gres, cpus = fields
        parsed = parse_gres(gres)
        if parsed is None:
            continue
        gpu_type, gpu_total = parsed
        cpu_parts = cpus.split("/")
        if len(cpu_parts) != 4:
            continue
        cpu_alloc, cpu_idle, cpu_other, cpu_total = map(int, cpu_parts)
        # sinfo can print repeated lines for a node; keep the first equivalent row.
        nodes.setdefault(
            name,
            NodeInfo(
                name=name,
                state=state,
                gpu_type=gpu_type,
                gpu_total=gpu_total,
                cpu_alloc=cpu_alloc,
                cpu_idle=cpu_idle,
                cpu_other=cpu_other,
                cpu_total=cpu_total,
            ),
        )
    return nodes


def get_jobs() -> list[JobInfo]:
    out = run_cmd(["squeue", "-h", "-o", "%i|%P|%j|%T|%M|%l|%N|%b"])
    jobs: list[JobInfo] = []
    for line in out.splitlines():
        fields = line.split("|")
        if len(fields) != 8:
            continue
        job_id, partition, name, state, runtime, timelimit, node, tres = fields
        gpu_type, gpu_count = parse_tres_gpu(tres)
        jobs.append(
            JobInfo(
                job_id=job_id,
                partition=partition,
                name=name,
                state=state,
                runtime_s=parse_duration_s(runtime),
                timelimit_s=parse_duration_s(timelimit),
                node=node,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
            )
        )
    return jobs


def node_used_gpus(node: NodeInfo, jobs: Iterable[JobInfo]) -> int:
    used = 0
    for job in jobs:
        if job.state != "RUNNING" or job.node != node.name:
            continue
        if job.gpu_count <= 0:
            continue
        if job.gpu_type is None or job.gpu_type == node.gpu_type:
            used += job.gpu_count
    return min(used, node.gpu_total)


def release_eta_for_needed(
    node: NodeInfo,
    jobs: Iterable[JobInfo],
    need_gpus: int,
    used_now: int,
) -> int | None:
    free_now = node.gpu_total - used_now
    if free_now >= need_gpus:
        return 0
    releases = []
    for job in jobs:
        if job.state != "RUNNING" or job.node != node.name or job.gpu_count <= 0:
            continue
        if job.gpu_type is not None and job.gpu_type != node.gpu_type:
            continue
        releases.append((job.remaining_s, job.gpu_count))
    releases.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**18))
    free = free_now
    last_eta: int | None = None
    for eta, count in releases:
        if eta is None:
            continue
        free += count
        last_eta = eta
        if free >= need_gpus:
            return last_eta
    return None


def score_node(
    node: NodeInfo,
    jobs: list[JobInfo],
    need_gpus: int,
    soon_s: int,
) -> tuple[int, int, int, str]:
    used = node_used_gpus(node, jobs)
    free = node.gpu_total - used
    eta = release_eta_for_needed(node, jobs, need_gpus, used)
    if free >= need_gpus:
        return (0, -free, node.cpu_alloc, "use_now")
    if eta is not None and eta <= soon_s:
        return (1, eta, -free, "wait_short")
    if eta is not None:
        return (2, eta, -free, "wait_long")
    return (3, 10**18, -free, "avoid")


def build_recommendations(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    nodes = get_nodes()
    jobs = get_jobs()
    wanted_type = normalize_gpu_type(args.type) if args.type else None
    candidates = []
    all_rows = []
    for node in nodes.values():
        if wanted_type and node.gpu_type != wanted_type:
            continue
        if args.exclude and node.name in args.exclude:
            continue
        if node.state.lower() in {"down", "drain", "drng", "fail", "maint"}:
            continue
        used = node_used_gpus(node, jobs)
        free = node.gpu_total - used
        eta = release_eta_for_needed(node, jobs, args.gpus, used)
        score = score_node(node, jobs, args.gpus, args.soon_minutes * 60)
        row = {
            "node": node,
            "used": used,
            "free": free,
            "eta": eta,
            "score": score,
            "jobs": [
                job
                for job in jobs
                if job.state == "RUNNING" and job.node == node.name and job.gpu_count > 0
            ],
        }
        all_rows.append(row)
        if score[3] in {"use_now", "wait_short"}:
            candidates.append(row)
    all_rows.sort(key=lambda r: r["score"])
    candidates.sort(key=lambda r: r["score"])
    return candidates, all_rows


def print_table(rows: list[dict], limit: int) -> None:
    print("node gpu total used free state eta decision running_jobs")
    for row in rows[:limit]:
        node: NodeInfo = row["node"]
        decision = row["score"][3]
        job_bits = []
        for job in row["jobs"]:
            job_bits.append(
                f"{job.job_id}:{job.name}:gpu{job.gpu_count}:run={format_duration(job.runtime_s)}:left={format_duration(job.remaining_s)}"
            )
        print(
            f"{node.name} {node.gpu_type} {node.gpu_total} {row['used']} {row['free']} "
            f"{node.state} {format_duration(row['eta'])} {decision} "
            + (";".join(job_bits) if job_bits else "-")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4, help="GPUs required by the job.")
    parser.add_argument(
        "--type",
        default=None,
        help="GPU type filter, e.g. 6000, a6000, a100, 3090. Omit to consider all.",
    )
    parser.add_argument(
        "--soon-minutes",
        type=int,
        default=60,
        help="Treat a node as worth waiting for if enough GPUs free within this many minutes.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Node names to exclude, e.g. elm66.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Rows to print in the detail table.",
    )
    parser.add_argument(
        "--allow-wait-long",
        action="store_true",
        help="If no immediate/soon candidate exists, recommend the shortest long wait.",
    )
    args = parser.parse_args()

    candidates, all_rows = build_recommendations(args)
    chosen = candidates[0] if candidates else (all_rows[0] if args.allow_wait_long and all_rows else None)

    if chosen is None:
        print("RECOMMEND no fixed node")
        print(f"SBATCH --gres=gpu:{args.type + ':' if args.type else ''}{args.gpus}")
        print("reason: no suitable node found; let Slurm place it globally")
        print()
        print_table(all_rows, args.limit)
        return 1

    node: NodeInfo = chosen["node"]
    decision = chosen["score"][3]
    if decision == "use_now":
        action = "use fixed node now"
    elif decision == "wait_short":
        action = f"queue fixed node; enough GPUs expected in {format_duration(chosen['eta'])}"
    else:
        action = f"long wait only; enough GPUs expected in {format_duration(chosen['eta'])}"

    print(f"RECOMMEND {node.name} gpu:{node.gpu_type}:{args.gpus}")
    print(f"ACTION {action}")
    print(f"SBATCH --nodelist={node.name} --gres=gpu:{node.gpu_type}:{args.gpus}")
    print()
    print_table(all_rows, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
