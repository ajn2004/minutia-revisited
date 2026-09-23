"""End-to-end tensor-native MINuTIA iteration benchmark.

This measures the complete accelerator-resident path, not a fused kernel.
Use ``--device cuda`` on CUDA or ROCm PyTorch installations.
"""

from __future__ import annotations

import argparse
import time

import torch

from minutia.detector import CanonicalDetector
from minutia.pipeline.batched import run_iteration_batched
from minutia.sim import simulate_movie
from minutia.training import TensorReplayBuffer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    movie = simulate_movie(
        n_frames=args.frames,
        height=args.height,
        width=args.width,
        molecules_per_frame=max(1, args.width // 32),
        photons=1200.0,
        poisson=True,
        seed=9,
    )
    frames = movie.frames.to(device=device, dtype=torch.float32)
    detector = CanonicalDetector().to(device)
    replay = TensorReplayBuffer(capacity=20000, device=device, seed=9)
    def run() -> object:
        return run_iteration_batched(
            frames, detector, replay, train_steps=1, localization_chunk_size=args.chunk_size
        )
    for _ in range(args.warmup):
        run()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.repeats):
        run()
    synchronize(device)
    elapsed = (time.perf_counter() - start) / args.repeats
    print(f"device={device} batch={args.frames} shape={tuple(frames.shape)} seconds={elapsed:.6f}")
    print(f"iterations_per_second={1.0 / elapsed:.3f} replay={replay.num_examples}")


if __name__ == "__main__":
    main()
