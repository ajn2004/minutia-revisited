"""Benchmark reference, batched CPU, and optional accelerator localization."""

import argparse
import time

import torch

from minutia.localization import localize_candidates, localize_candidates_batched


def measure(function: object, synchronize: bool, repeats: int) -> float:
    callable_function = function  # type: ignore[assignment]
    if synchronize:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        callable_function()
    if synchronize:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        print(f"device={device}")
        frames = torch.full((4, 128, 128), 5.0, device=device)
        for count in (1, 10, 100, 1000):
            if count > args.max_candidates:
                continue
            x = torch.arange(count, device=device) % 120 + 4
            y = (torch.arange(count, device=device) * 7) % 120 + 4
            frame = torch.arange(count, device=device) % 4
            candidates = torch.stack((frame, x, y, torch.ones_like(x)), dim=1).float()
            sync = device.type == "cuda"
            reference = measure(
                lambda: localize_candidates(frames, candidates, iterations=5), sync, args.repeats
            )
            batched = measure(
                lambda: localize_candidates_batched(frames, candidates, iterations=5),
                sync,
                args.repeats,
            )
            print(f"candidates={count} reference_s={reference:.6f} batched_s={batched:.6f}")


if __name__ == "__main__":
    main()
