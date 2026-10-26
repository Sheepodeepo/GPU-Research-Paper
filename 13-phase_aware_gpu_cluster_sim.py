#!/usr/bin/env python3
"""
Phase-aware heterogeneous GPU cluster simulator
================================================

Self-contained, standard-library Python model for an EE 295 research paper.
The simulator compares four policies on the same generated workload trace:

  backfill   : short jobs may use an idle GPU before the queue head
  workload   : workload/GPU matching with maximum frequency
  proposed   : jointly selects queue position, GPU type, and DVFS state

The proposed policy models a cluster-level phase-aware decision.  At every
job arrival, phase completion, or GPU completion, it evaluates feasible
(job, GPU, frequency) candidates under a fixed cluster power budget.  Queue
position is represented by the order in which candidates are selected for
currently available GPUs.  Jobs may change frequency and GPU between phases;
DVFS and migration overheads are modeled explicitly.

Run examples:
    python phase_aware_gpu_cluster_sim.py
    python phase_aware_gpu_cluster_sim.py --jobs 100 --seed 7 --plot
    python phase_aware_gpu_cluster_sim.py --policy proposed --csv results.csv

The numerical hardware/workload values are illustrative model parameters, not
physical measurements. Replace them with calibrated values from your sources.
Only Python's standard library is required. Matplotlib is optional for plots.
"""

from __future__ import annotations

import argparse
import csv
import copy
import heapq
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------- Model parameters -----------------------------

POWER_BUDGET_W = 330.0
LATENCY_SLACK = 0.05  # 5% allowed beyond the job's modeled deadline
DVFS_TRANSITION_S = 0.025
MIGRATION_S = 0.050
MIGRATION_J = 2.0


@dataclass
class Phase:
    kind: str
    work: float


@dataclass
class Job:
    jid: int
    arrival: float
    phases: List[Phase]
    deadline: float
    current_phase: int = 0
    first_start: Optional[float] = None
    completion: Optional[float] = None
    queue_time: float = 0.0
    queue_entered: Optional[float] = None
    energy_j: float = 0.0
    dvfs_transitions: int = 0
    migrations: int = 0
    last_gpu: Optional[str] = None
    last_freq: Optional[float] = None

    @property
    def done(self) -> bool:
        return self.current_phase >= len(self.phases)


@dataclass
class GPU:
    gid: str
    kind: str
    performance: float
    memory_factor: float
    dynamic_power: float
    idle_power: float
    frequencies: Tuple[float, ...] = (0.60, 0.80, 1.00)
    busy_job: Optional[int] = None
    busy_power: float = 0.0
    busy_since: Optional[float] = None
    busy_time: float = 0.0

    def free(self) -> bool:
        return self.busy_job is None


@dataclass
class Event:
    time: float
    sequence: int
    kind: str
    payload: object = field(compare=False)

    def __lt__(self, other: "Event") -> bool:
        return (self.time, self.sequence) < (other.time, other.sequence)


def make_gpus() -> List[GPU]:
    # Relative performance and power values are intentionally transparent.
    return [
        GPU("A100-0", "A100", performance=4.00, memory_factor=1.00,
            dynamic_power=230.0, idle_power=35.0),
        GPU("A100-1", "A100", performance=4.00, memory_factor=1.00,
            dynamic_power=230.0, idle_power=35.0),
        GPU("T4-0", "T4", performance=1.50, memory_factor=0.65,
            dynamic_power=55.0, idle_power=12.0),
        GPU("T4-1", "T4", performance=1.50, memory_factor=0.65,
            dynamic_power=55.0, idle_power=12.0),
    ]


# ------------------------------ Workload model ------------------------------

def generate_jobs(n: int, seed: int) -> List[Job]:
    """Generate one repeatable trace shared by every policy."""
    rng = random.Random(seed)
    jobs: List[Job] = []
    now = 0.0
    kinds = ("compute", "memory", "mixed")

    for jid in range(n):
        # Mean inter-arrival time is about 16.7 s.  This creates a moderate
        # load by default; increase the rate for an overload experiment.
        now += rng.expovariate(0.06) if jid else 0.0
        workload = rng.choice(kinds)
        phase_count = rng.choice((2, 3, 4))
        phases: List[Phase] = []

        for p in range(phase_count):
            # Mostly preserve the job class but include phase transitions.
            if workload == "compute":
                phase_kind = rng.choices(
                    ["compute", "mixed", "memory"], [0.65, 0.25, 0.10]
                )[0]
            elif workload == "memory":
                phase_kind = rng.choices(
                    ["memory", "mixed", "compute"], [0.65, 0.25, 0.10]
                )[0]
            else:
                phase_kind = rng.choice(kinds)
            phases.append(Phase(phase_kind, rng.uniform(18.0, 42.0)))

        # Deadline is based on an optimistic single-A100 estimate.  The
        # multiplier leaves room for queueing but still makes overload and
        # poor placement visible as deadline violations.
        optimistic = sum(p.work / 4.0 for p in phases)
        deadline = now + optimistic * rng.uniform(2.80, 3.80)
        jobs.append(Job(jid, now, phases, deadline))

    return jobs


# ------------------------------ GPU calculations ----------------------------

def frequency_factor(freq: float) -> float:
    return freq


def phase_throughput(phase: Phase, gpu: GPU, freq: float) -> float:
    """Relative work units per second for a phase/GPU/frequency pair."""
    f = frequency_factor(freq)
    if phase.kind == "compute":
        sensitivity = f
    elif phase.kind == "memory":
        # Memory-bound phases gain less from core-frequency increases.
        sensitivity = 0.72 + 0.28 * f
    else:
        sensitivity = 0.84 + 0.16 * f
    return max(0.01, gpu.performance * gpu.memory_factor * sensitivity)


def phase_power(phase: Phase, gpu: GPU, freq: float) -> float:
    """Modeled instantaneous GPU power in watts."""
    activity = {"compute": 1.00, "memory": 0.78, "mixed": 0.90}[phase.kind]
    dynamic = gpu.dynamic_power * activity * (0.42 + 0.58 * freq ** 3)
    return gpu.idle_power + dynamic


def phase_runtime(phase: Phase, gpu: GPU, freq: float) -> float:
    return phase.work / phase_throughput(phase, gpu, freq)


def phase_energy(phase: Phase, gpu: GPU, freq: float) -> float:
    return phase_power(phase, gpu, freq) * phase_runtime(phase, gpu, freq)


# ---------------------------- Simulation engine -----------------------------

class Simulator:
    def __init__(self, jobs: List[Job], policy: str,
                 power_budget: float = POWER_BUDGET_W,
                 latency_slack: float = LATENCY_SLACK):
        self.jobs = copy.deepcopy(jobs)
        self.policy = policy
        self.power_budget = power_budget
        self.latency_slack = latency_slack
        self.gpus = make_gpus()
        self.job_by_id = {j.jid: j for j in self.jobs}
        self.waiting: List[Job] = []
        self.events: List[Event] = []
        self.event_seq = 0
        self.time = 0.0
        self.total_energy = 0.0
        self.power_violations = 0
        self.schedule_decisions = 0
        self.schedule_time = 0.0
        self.power_trace: List[Tuple[float, float]] = []

    def push(self, time: float, kind: str, payload: object) -> None:
        heapq.heappush(self.events, Event(time, self.event_seq, kind, payload))
        self.event_seq += 1

    def current_power(self) -> float:
        return sum(g.busy_power for g in self.gpus)

    def advance_energy(self, new_time: float) -> None:
        dt = max(0.0, new_time - self.time)
        power = self.current_power()
        self.total_energy += power * dt
        self.power_trace.append((new_time, power))
        for gpu in self.gpus:
            if not gpu.free():
                gpu.busy_time += dt
        self.time = new_time

    def run(self) -> Dict[str, object]:
        for job in self.jobs:
            self.push(job.arrival, "arrival", job.jid)

        while self.events:
            event = heapq.heappop(self.events)
            self.advance_energy(event.time)

            if event.kind == "arrival":
                job = self.job_by_id[event.payload]
                job.queue_entered = self.time
                self.waiting.append(job)
            elif event.kind == "phase_done":
                self.finish_phase(event.payload)

            self.dispatch()

        self.advance_energy(self.time)
        return self.metrics()

    def finish_phase(self, payload: Tuple[int, str, float]) -> None:
        jid, gid, phase_energy_j = payload
        job = self.job_by_id[jid]
        gpu = next(g for g in self.gpus if g.gid == gid)
        job.energy_j += phase_energy_j
        gpu.busy_job = None
        gpu.busy_power = 0.0
        gpu.busy_since = None
        job.current_phase += 1

        if job.done:
            job.completion = self.time
        else:
            job.queue_entered = self.time
            self.waiting.append(job)

    def dispatch(self) -> None:
        start_clock = self.time
        while self.waiting:
            free_gpus = [g for g in self.gpus if g.free()]
            if not free_gpus:
                break
            candidate = self.choose_candidate(self.waiting, free_gpus)
            if candidate is None:
                # No feasible assignment under the cluster power budget.
                break
            job, gpu, freq, runtime, power, energy, score = candidate
            self.waiting.remove(job)
            self.start_phase(job, gpu, freq, runtime, power, energy)
            self.schedule_decisions += 1
        self.schedule_time += self.time - start_clock

    def start_phase(self, job: Job, gpu: GPU, freq: float, runtime: float,
                    power: float, energy: float) -> None:
        if job.first_start is None:
            job.first_start = self.time
        if job.queue_entered is not None:
            job.queue_time += max(0.0, self.time - job.queue_entered)
        if job.last_freq is not None and abs(job.last_freq - freq) > 1e-9:
            runtime += DVFS_TRANSITION_S
            energy += power * DVFS_TRANSITION_S
            job.dvfs_transitions += 1
        if job.last_gpu is not None and job.last_gpu != gpu.gid:
            runtime += MIGRATION_S
            energy += MIGRATION_J
            job.migrations += 1
        job.last_freq = freq
        job.last_gpu = gpu.gid
        gpu.busy_job = job.jid
        gpu.busy_power = power
        gpu.busy_since = self.time
        self.push(self.time + runtime, "phase_done",
                  (job.jid, gpu.gid, energy))

    def remaining_work(self, job: Job) -> float:
        return sum(p.work for p in job.phases[job.current_phase:])

    def estimate_remaining_time(self, job: Job, gpu: GPU, freq: float) -> float:
        return sum(phase_runtime(p, gpu, freq)
                   for p in job.phases[job.current_phase:])

    def candidate_values(self, job: Job, gpu: GPU, freq: float):
        phase = job.phases[job.current_phase]
        runtime = phase_runtime(phase, gpu, freq)
        energy = phase_energy(phase, gpu, freq)
        power = phase_power(phase, gpu, freq)
        if job.last_freq is not None and abs(job.last_freq - freq) > 1e-9:
            runtime += DVFS_TRANSITION_S
            energy += power * DVFS_TRANSITION_S
        if job.last_gpu is not None and job.last_gpu != gpu.gid:
            runtime += MIGRATION_S
            energy += MIGRATION_J
        return runtime, power, energy

    def choose_candidate(self, waiting: List[Job], free_gpus: List[GPU]):
        feasible = []
        current_power = self.current_power()

        for job in waiting:
            for gpu in free_gpus:
                for freq in gpu.frequencies:
                    runtime, power, energy = self.candidate_values(job, gpu, freq)
                    if current_power + power > self.power_budget + 1e-9:
                        continue
                    finish = self.time + runtime
                    lateness = max(0.0, finish - job.deadline)
                    queue_delay = (self.time - job.queue_entered
                                   if job.queue_entered is not None else 0.0)
                    edp = energy * max(runtime, 1e-9)
                    feasible.append((job, gpu, freq, runtime, power,
                                     energy, lateness, queue_delay, edp))

        if not feasible:
            return None

        if self.policy == "backfill":
            # Prefer short jobs, but preserve the queue head when it fits.
            head = min(feasible, key=lambda x: (x[0].arrival, x[0].jid))[0]
            head_options = [x for x in feasible if x[0].jid == head.jid]
            if head_options:
                head_best = min(head_options, key=lambda x: x[3])
                short = [x for x in feasible if x[3] <= head_best[3] * 0.55]
                feasible = short or head_options
            feasible.sort(key=lambda x: (x[3], x[0].arrival, x[0].jid))
        elif self.policy == "workload":
            # Match phase type to GPU capability, but do not lower frequency.
            feasible = [x for x in feasible if abs(x[2] - 1.0) < 1e-9] or feasible
            feasible.sort(key=lambda x: (x[3], x[0].arrival, x[0].jid))
        elif self.policy == "dvfs":
            # Head-of-line scheduling with energy-minimizing DVFS.
            head = min(feasible, key=lambda x: (x[0].arrival, x[0].jid))[0]
            feasible = [x for x in feasible if x[0].jid == head.jid]
            feasible.sort(key=lambda x: (x[5], x[3]))
        else:  # proposed
            # Joint queue/GPU/frequency score. Normalize terms so no one unit
            # dominates the others. Hard latency violations receive a penalty.
            def score(x):
                job, gpu, freq, runtime, power, energy, late, qdelay, edp = x
                normalized_energy = energy / 100.0
                normalized_edp = edp / 1000.0
                normalized_queue = qdelay / 10.0
                violation = late / max(job.deadline - job.arrival, 1.0)
                phase_bonus = 0.0
                phase = job.phases[job.current_phase].kind
                if phase == "compute" and gpu.kind == "A100":
                    phase_bonus -= 0.08
                if phase == "memory" and gpu.kind == "T4":
                    phase_bonus -= 0.03
                return (0.38 * normalized_energy
                        + 0.22 * normalized_edp
                        + 0.12 * normalized_queue
                        + 0.08 * (runtime / 100.0)
                        + 4.0 * violation
                        + phase_bonus)
            feasible.sort(key=score)

        chosen = feasible[0]
        job, gpu, freq, runtime, power, energy = chosen[:6]
        return job, gpu, freq, runtime, power, energy, 0.0

    def metrics(self) -> Dict[str, object]:
        completed = [j for j in self.jobs if j.completion is not None]
        latencies = [j.completion - j.arrival for j in completed]
        deadline_violations = [
            j for j in completed
            if j.completion > j.deadline * (1.0 + self.latency_slack)
        ]
        makespan = max((j.completion for j in completed), default=0.0)
        total_work = sum(p.work for j in self.jobs for p in j.phases)
        throughput = total_work / makespan if makespan else 0.0
        utilization = (sum(g.busy_time for g in self.gpus)
                       / (len(self.gpus) * makespan)
                       if makespan else 0.0)
        avg_power = self.total_energy / makespan if makespan else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        return {
            "policy": self.policy,
            "jobs": len(self.jobs),
            "completed": len(completed),
            "total_energy_j": round(self.total_energy, 3),
            "avg_power_w": round(avg_power, 3),
            "avg_latency_s": round(avg_latency, 3),
            "p95_latency_s": round(percentile(latencies, 0.95), 3),
            "avg_queue_delay_s": round(statistics.mean(
                [j.queue_time for j in completed]) if completed else 0.0, 3),
            "throughput_work_per_s": round(throughput, 3),
            "gpu_utilization": round(utilization, 4),
            "edp": round(self.total_energy * avg_latency, 3),
            "deadline_violations": len(deadline_violations),
            "deadline_violation_rate": round(
                len(deadline_violations) / len(self.jobs), 4) if self.jobs else 0.0,
            "dvfs_transitions": sum(j.dvfs_transitions for j in self.jobs),
            "migrations": sum(j.migrations for j in self.jobs),
            "scheduler_decisions": self.schedule_decisions,
            "power_budget_w": self.power_budget,
            "latency_slack": self.latency_slack,
            "makespan_s": round(makespan, 3),
        }


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


# ---------------------------------- Output ----------------------------------

def print_results(results: List[Dict[str, object]]) -> None:
    columns = [
        ("policy", "Policy"),
        ("total_energy_j", "Energy (J)"),
        ("avg_latency_s", "Avg latency (s)"),
        ("avg_queue_delay_s", "Queue delay (s)"),
        ("throughput_work_per_s", "Throughput"),
        ("gpu_utilization", "Utilization"),
        ("edp", "EDP"),
        ("deadline_violation_rate", "Violation rate"),
    ]
    widths = [max(len(title), *(len(str(r[key])) for r in results))
              for key, title in columns]
    header = "  ".join(title.ljust(width) for (_, title), width in zip(columns, widths))
    print("\n" + header)
    print("  ".join("-" * width for width in widths))
    for result in results:
        print("  ".join(str(result[key]).ljust(width)
                       for (key, _), width in zip(columns, widths)))


def save_csv(results: List[Dict[str, object]], filename: str) -> None:
    if not results:
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def plot_results(results: List[Dict[str, object]]) -> None:
    try:
        import matplotlib
        # Keep text editable/sharp in vector exports when possible.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: install matplotlib with `python -m pip install matplotlib`.")
        return

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    labels = [r["policy"] for r in results]
    energy = [r["total_energy_j"] for r in results]
    latency = [r["avg_latency_s"] for r in results]
    violations = [100 * r["deadline_violation_rate"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(labels, energy, color="#4472C4")
    axes[0].set_title("Total energy")
    axes[0].set_ylabel("J")
    axes[1].bar(labels, latency, color="#70AD47")
    axes[1].set_title("Average latency")
    axes[1].set_ylabel("seconds")
    axes[2].bar(labels, violations, color="#ED7D31")
    axes[2].set_title("Latency/deadline violations")
    axes[2].set_ylabel("percent of jobs")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Phase-aware heterogeneous GPU cluster simulation")
    fig.tight_layout()

    # Export vector formats for IEEE/Word insertion, plus a high-resolution
    # raster fallback. PDF is the preferred Word-compatible vector format;
    # SVG is also retained for editing or use in other publishing tools.
    output_files = [
        ("gpu_cluster_results.pdf", {"bbox_inches": "tight"}),
        ("gpu_cluster_results.svg", {"bbox_inches": "tight"}),
        ("gpu_cluster_results.png", {"dpi": 300, "bbox_inches": "tight"}),
    ]
    for filename, options in output_files:
        fig.savefig(filename, **options)
        print(f"Saved plot: {filename}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=80,
                        help="number of jobs in the generated trace")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed; use the same seed across policies")
    parser.add_argument("--power-budget", type=float, default=POWER_BUDGET_W,
                        help="cluster power budget in watts")
    parser.add_argument("--latency-slack", type=float, default=LATENCY_SLACK,
                        help="allowed fractional deadline slack, e.g. 0.05")
    parser.add_argument("--policy", choices=["all", "backfill",
                                              "workload", "dvfs", "proposed"],
                        default="all")
    parser.add_argument("--csv", default="",
                        help="optional output CSV filename")
    parser.add_argument("--json", default="",
                        help="optional output JSON filename")
    parser.add_argument("--plot", action="store_true",
                        help=("save gpu_cluster_results.pdf, .svg, and a "
                              "300-dpi .png fallback; requires matplotlib"))
    args = parser.parse_args()

    trace = generate_jobs(args.jobs, args.seed)
    policies = ["backfill", "workload", "dvfs", "proposed"]
    if args.policy != "all":
        policies = [args.policy]

    results = []
    for policy in policies:
        result = Simulator(trace, policy, args.power_budget,
                           args.latency_slack).run()
        results.append(result)

    print_results(results)
    if args.csv:
        save_csv(results, args.csv)
        print(f"Saved results: {args.csv}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results: {args.json}")
    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
