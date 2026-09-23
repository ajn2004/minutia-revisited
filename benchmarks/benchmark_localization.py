"""Benchmark reference and batched localization with trustworthy device timing.

The default benchmark measures localization only.  Frames and candidates are
created before the timed region so allocation and candidate construction do
not obscure the comparison.  Use ``--end-to-end`` when those costs are part of
the question being measured.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from minutia.localization import localize_candidates, localize_candidates_batched

DEFAULT_REFERENCE_COUNTS = (1, 10, 100)
DEFAULT_BATCHED_COUNTS = (1, 10, 100, 1_000, 10_000)


def _synchronize(device: torch.device) -> None:
    """Wait for work on an accelerator, including AMD ROCm devices."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    function: Callable[[], object],
    device: torch.device,
    repeats: int,
    warmup: int,
) -> list[float]:
    """Return one independently synchronized sample per measured invocation."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")

    # Warm-up is deliberately done for every implementation/device/count pair.
    for _ in range(warmup):
        function()
    _synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        function()
        # The stop time must follow synchronization, otherwise it only measures
        # kernel launch latency on an asynchronous accelerator.
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return samples


def _summary(samples: Sequence[float]) -> tuple[float, float, float, float]:
    """Return p25, median, p75, and minimum in seconds."""
    ordered = sorted(samples)
    return (
        statistics.quantiles(ordered, n=4, method="inclusive")[0]
        if len(ordered) > 1
        else ordered[0],
        statistics.median(ordered),
        statistics.quantiles(ordered, n=4, method="inclusive")[2]
        if len(ordered) > 1
        else ordered[0],
        min(ordered),
    )


def _format_result(label: str, samples: Sequence[float], count: int) -> str:
    p25, median, p75, minimum = _summary(samples)
    # Throughput is reported from the same robust statistic as the timing.
    rate = count / median if median > 0 else float("inf")
    return (
        f"{label}_s(p25/median/p75/min)="
        f"{p25:.6f}/{median:.6f}/{p75:.6f}/{minimum:.6f} "
        f"{label}_candidates_per_sec={rate:.2f}"
    )


def _make_candidates(count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(count, device=device)
    x = index % 120 + 4
    y = (index * 7) % 120 + 4
    frame = index % 4
    return torch.stack((frame, x, y, torch.ones_like(x)), dim=1).to(dtype)


def _report_environment(device: torch.device, dtype: torch.dtype) -> None:
    print(f"torch_version={torch.__version__}")
    print(f"hip_version={torch.version.hip or 'unavailable'}")
    print(f"cuda_version={torch.version.cuda or 'unavailable'}")
    print(f"device={device} device_name=", end="")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
        properties = torch.cuda.get_device_properties(device)
        print(
            "device_properties="
            f"total_memory={properties.total_memory}, "
            f"multi_processor_count={properties.multi_processor_count}, "
            f"compute_capability={properties.major}.{properties.minor}"
        )
    else:
        print("CPU")
        print("device_properties=unavailable")
    print(f"dtype={dtype}")


def _counts(value: Sequence[int] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    selected = tuple(value) if value else default
    if any(count < 1 for count in selected):
        raise ValueError("candidate counts must be positive")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts", nargs="+", type=int, help="use these counts for both implementations"
    )
    parser.add_argument("--reference-counts", nargs="+", type=int)
    parser.add_argument("--batched-counts", nargs="+", type=int)
    parser.add_argument("--max-candidates", type=int, help="compatibility cap for all counts")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batched-only", action="store_true")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    common = _counts(args.counts, DEFAULT_BATCHED_COUNTS)
    reference_counts = (
        common
        if args.counts
        else _counts(args.reference_counts, DEFAULT_REFERENCE_COUNTS)
    )
    batched_counts = common if args.counts else _counts(args.batched_counts, DEFAULT_BATCHED_COUNTS)
    if args.max_candidates is not None:
        reference_counts = tuple(c for c in reference_counts if c <= args.max_candidates)
        batched_counts = tuple(c for c in batched_counts if c <= args.max_candidates)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        _report_environment(device, dtype)
        for count in sorted(set(reference_counts) | set(batched_counts)):
            frames = torch.full((4, 128, 128), 5.0, dtype=dtype, device=device)
            candidates = _make_candidates(count, device, dtype)

            def run_reference() -> object:
                if args.end_to_end:
                    local_frames = torch.full((4, 128, 128), 5.0, dtype=dtype, device=device)
                    local_candidates = _make_candidates(count, device, dtype)
                    return localize_candidates(
                        local_frames, local_candidates, iterations=args.iterations
                    )
                return localize_candidates(frames, candidates, iterations=args.iterations)

            def run_batched() -> object:
                if args.end_to_end:
                    local_frames = torch.full((4, 128, 128), 5.0, dtype=dtype, device=device)
                    local_candidates = _make_candidates(count, device, dtype)
                    return localize_candidates_batched(
                        local_frames, local_candidates, iterations=args.iterations
                    )
                return localize_candidates_batched(frames, candidates, iterations=args.iterations)

            parts = [f"device={device} candidates={count}"]
            if not args.batched_only and count in reference_counts:
                parts.append(
                    _format_result(
                        "reference",
                        measure(run_reference, device, args.repeats, args.warmup),
                        count,
                    )
                )
            if count in batched_counts:
                parts.append(
                    _format_result(
                        "batched",
                        measure(run_batched, device, args.repeats, args.warmup),
                        count,
                    )
                )
            if len(parts) > 1:
                print(" ".join(parts))


if __name__ == "__main__":
    main()
