"""Schedule jobs by deadline and inspect their lateness."""

import math
from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Job:
    name: str
    processing_time: float
    deadline: float


@dataclass(frozen=True)
class ScheduledJob:
    job: Job
    completion_time: float
    lateness: float


@dataclass(frozen=True)
class Schedule:
    jobs: tuple
    maximum_lateness: float


def _check_jobs(jobs):
    jobs = list(jobs)
    names = set()

    for job in jobs:
        if not isinstance(job, Job):
            raise ValueError("jobs need to be Job objects")
        if not isinstance(job.name, str) or not job.name or job.name in names:
            raise ValueError("job names need to be unique and non-empty")
        if (
            isinstance(job.processing_time, bool)
            or not isinstance(job.processing_time, (int, float))
            or not math.isfinite(job.processing_time)
            or job.processing_time <= 0
        ):
            raise ValueError("processing times need to be finite and positive")
        if (
            isinstance(job.deadline, bool)
            or not isinstance(job.deadline, (int, float))
            or not math.isfinite(job.deadline)
        ):
            raise ValueError("deadlines need to be finite numbers")
        names.add(job.name)

    return jobs


def evaluate_order(jobs):
    """Calculate completion times and lateness for one chosen order."""
    jobs = _check_jobs(jobs)
    time = 0.0
    scheduled = []

    for job in jobs:
        time += job.processing_time
        scheduled.append(
            ScheduledJob(
                job=job,
                completion_time=time,
                lateness=time - job.deadline,
            )
        )

    maximum_lateness = max(
        (item.lateness for item in scheduled),
        default=0.0,
    )
    return Schedule(tuple(scheduled), maximum_lateness)


def earliest_deadline_schedule(jobs):
    """Put the jobs in increasing deadline order."""
    jobs = _check_jobs(jobs)
    ordered = sorted(jobs, key=lambda job: (job.deadline, job.name))
    return evaluate_order(ordered)


def brute_force_schedule(jobs):
    """Check every order, only as a comparison for small inputs."""
    jobs = _check_jobs(jobs)
    if len(jobs) > 9:
        raise ValueError("brute force is only meant for at most nine jobs")
    if not jobs:
        return evaluate_order([])

    best = None
    for order in permutations(jobs):
        candidate = evaluate_order(order)
        if best is None or candidate.maximum_lateness < best.maximum_lateness:
            best = candidate

    return best


def _names(schedule):
    return [item.job.name for item in schedule.jobs]


def run_example():
    jobs = [
        Job("write", processing_time=4, deadline=4),
        Job("email", processing_time=1, deadline=2),
        Job("review", processing_time=2, deadline=7),
        Job("upload", processing_time=3, deadline=9),
    ]

    original = evaluate_order(jobs)
    greedy = earliest_deadline_schedule(jobs)
    exact = brute_force_schedule(jobs)

    print(f"Original order:          {_names(original)}")
    print(f"Original max lateness:   {original.maximum_lateness:.1f}")
    print(f"Deadline order:          {_names(greedy)}")
    print(f"Deadline max lateness:   {greedy.maximum_lateness:.1f}")
    print(f"Brute-force best value:  {exact.maximum_lateness:.1f}\n")
    print(" job       completion   lateness")
    for item in greedy.jobs:
        print(
            f" {item.job.name:8}   {item.completion_time:10.1f}"
            f"   {item.lateness:8.1f}"
        )


if __name__ == "__main__":
    run_example()
