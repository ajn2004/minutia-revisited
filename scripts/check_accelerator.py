"""Report the PyTorch accelerator environment and run a batched-localizer smoke test."""

import torch

from minutia.localization import localize_candidates_batched


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"device name: {torch.cuda.get_device_name(device)}")
    else:
        print("device name: CPU")
    for dtype in (torch.float32, torch.float64):
        frames = torch.ones((1, 16, 16), device=device, dtype=dtype)
        candidates = torch.tensor([[0, 8, 8, 1]], device=device, dtype=dtype)
        result = localize_candidates_batched(frames, candidates, iterations=2)
        print(
            f"{dtype}: parameters={tuple(result.parameters.shape)}, "
            f"valid={result.valid.tolist()}"
        )
    print("batched-localizer smoke test: PASS")


if __name__ == "__main__":
    main()
