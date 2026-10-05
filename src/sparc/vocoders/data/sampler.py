"""Infinite sampler whose stream is a pure function of ``(seed, epoch, position)``, so a run can resume exactly."""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

SampleKey = tuple[int, int, int]


class ResumableSampler(Sampler):
    """Yields ``(epoch, position, item_index)`` keys forever.

    Epoch ``e`` is the permutation ``default_rng([seed, e]).permutation(num_items)``. Rank ``r`` of ``w`` ranks takes
    entries ``r, r + w, r + 2w, ...`` of it, and the first ``num_items // w`` of those make up its epoch, so all ranks
    see the same number of samples per epoch. ``position`` is the index into the permutation (unique across ranks);
    the dataset keys its random draws on ``(seed, epoch, position)``.

    The stream carries no hidden state: ``iter(sampler)`` starts at the position set by ``set_samples_consumed`` or
    ``load_state_dict`` (the start of the stream by default) and every iterator replays the same keys. A
    ``DataLoader`` prefetches ahead of the training loop, so the training module records the number of samples
    actually consumed and restores it instead of reading the sampler.
    """

    def __init__(self, num_items: int, seed: int, rank: int = 0, world_size: int = 1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if num_items < world_size:
            raise ValueError(f"{num_items} items cannot be split over {world_size} ranks")
        self.num_items = int(num_items)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch_length = self.num_items // self.world_size
        self._start = 0
        self._cached_epoch = -1
        self._cached_perm = np.empty(0, dtype=np.int64)

    @property
    def samples_consumed(self) -> int:
        """Samples consumed over all ranks at the current start of the stream."""
        return self._start * self.world_size

    def set_samples_consumed(self, samples_consumed: int) -> None:
        """Moves the start of the stream to ``samples_consumed`` samples summed over all ranks."""
        if samples_consumed < 0 or samples_consumed % self.world_size:
            raise ValueError(
                f"samples_consumed={samples_consumed} must be a non-negative multiple of {self.world_size}"
            )
        self._start = samples_consumed // self.world_size

    def state_dict(self) -> dict[str, int]:
        """Epoch and number of samples this rank has already taken from it."""
        epoch, position = divmod(self._start, self.epoch_length)
        return {"epoch": epoch, "position": position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._start = int(state["epoch"]) * self.epoch_length + int(state["position"])

    def permutation(self, epoch: int) -> np.ndarray:
        """Item order of ``epoch`` over all ranks."""
        if epoch != self._cached_epoch:
            self._cached_perm = np.random.default_rng([self.seed, epoch]).permutation(self.num_items)
            self._cached_epoch = epoch
        return self._cached_perm

    def key_at(self, count: int) -> SampleKey:
        """Key of the ``count``-th sample (0-based) taken by this rank since the start of training."""
        epoch, k = divmod(count, self.epoch_length)
        position = k * self.world_size + self.rank
        return epoch, position, int(self.permutation(epoch)[position])

    def __iter__(self) -> Iterator[SampleKey]:
        return self._stream(self._start)

    def _stream(self, count: int) -> Iterator[SampleKey]:
        while True:
            yield self.key_at(count)
            count += 1
