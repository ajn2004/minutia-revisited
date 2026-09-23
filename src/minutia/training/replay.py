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


class TensorReplayBuffer:
    """Bounded replay memory whose records remain tensors on one device.

    This is deliberately separate from :class:`ReplayBuffer`: the latter is
    the readable scientific oracle and stores ``TrainingExample`` objects.
    Retention is performed with a device-side permutation and tensor indexing.
    """

    _fields = ("patches", "labels", "frame", "x", "y", "detector_score", "fit_parameters")

    def __init__(
        self, capacity: int = 10000, *, device: torch.device | str = "cpu", seed: int = 0
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self.patches = torch.empty((0, 7, 7), device=self.device)
        self.labels = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.frame = torch.empty((0,), dtype=torch.long, device=self.device)
        self.x = torch.empty((0,), dtype=torch.long, device=self.device)
        self.y = torch.empty((0,), dtype=torch.long, device=self.device)
        self.detector_score = torch.empty((0,), device=self.device)
        self.fit_parameters = torch.empty((0, 6), device=self.device)

    @property
    def num_examples(self) -> int:
        return self.patches.shape[0]

    def clear(self) -> None:
        keep = torch.empty((0,), dtype=torch.long, device=self.device)
        self._retain(keep)

    def add(
        self,
        patches: torch.Tensor,
        labels: torch.Tensor,
        frame: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        detector_score: torch.Tensor,
        fit_parameters: torch.Tensor,
    ) -> None:
        tensors = (patches, labels, frame, x, y, detector_score, fit_parameters)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all replay tensors must be on the configured device")
        n = patches.shape[0]
        if patches.ndim != 3 or patches.shape[1:] != (7, 7) or fit_parameters.shape != (n, 6):
            raise ValueError("patches must be [N,7,7] and fit_parameters must be [N,6]")
        if any(t.shape[0] != n for t in tensors):
            raise ValueError("replay fields must have the same first dimension")
        if n == 0:
            return
        self.patches = torch.cat((self.patches, patches.detach()), dim=0)
        self.labels = torch.cat((self.labels, labels.detach().to(torch.bool)), dim=0)
        self.frame = torch.cat((self.frame, frame.detach().to(torch.long)), dim=0)
        self.x = torch.cat((self.x, x.detach().to(torch.long)), dim=0)
        self.y = torch.cat((self.y, y.detach().to(torch.long)), dim=0)
        self.detector_score = torch.cat((self.detector_score, detector_score.detach()), dim=0)
        self.fit_parameters = torch.cat((self.fit_parameters, fit_parameters.detach()), dim=0)
        if self.num_examples > self.capacity:
            keep = torch.randperm(
                self.num_examples, device=self.device, generator=self._generator
            )[: self.capacity]
            self._retain(keep)

    def _retain(self, indices: torch.Tensor) -> None:
        self.patches = self.patches[indices]
        self.labels = self.labels[indices]
        self.frame = self.frame[indices]
        self.x = self.x[indices]
        self.y = self.y[indices]
        self.detector_score = self.detector_score[indices]
        self.fit_parameters = self.fit_parameters[indices]

    def training_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_examples == 0:
            raise ValueError("cannot materialize an empty replay buffer")
        return self.patches, self.labels.to(dtype=self.patches.dtype)
