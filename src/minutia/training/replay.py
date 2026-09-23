"""Modern bounded replay memory for fit-guided detector training."""

from dataclasses import dataclass

import torch


@dataclass
class TrainingExample:
    patch: torch.Tensor
    label: bool
    frame: int
    x: int
    y: int
    score: float
    fit_parameters: torch.Tensor


class ReplayBuffer:
    """Uniformly downsampled bounded memory.

    This is a modern engineering policy, not a claim to reproduce the
    historical example-retention behavior. The legacy implementation's
    retention/discarding details are documented separately.
    """
    def __init__(self, capacity: int = 10000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.examples: list[TrainingExample] = []
        self.generator = torch.Generator().manual_seed(seed)

    def add(self, examples: list[TrainingExample]) -> None:
        self.examples.extend(examples)
        if len(self.examples) > self.capacity:
            keep = torch.randperm(len(self.examples), generator=self.generator)[
                : self.capacity
            ].tolist()
            self.examples = [self.examples[i] for i in keep]

    def tensors(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.examples:
            raise ValueError("cannot materialize an empty replay buffer")
        patches = torch.stack([e.patch for e in self.examples]).to(device=device)
        labels = torch.tensor([e.label for e in self.examples], dtype=patches.dtype, device=device)
        return patches, labels
