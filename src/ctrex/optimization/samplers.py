"""Ordered-subset (OS) sampling of sinogram projection indices for CT reconstruction.

Defines `SinogramSampler` and its subclasses, which decide how the full set of
projection angles is split into subsets that `CTReconstruction` iterates over one at a
time (one training step per subset, one epoch per full sweep of subsets), plus a
trivial single-item dataset used to drive Lightning's validation loop.
"""

import random

import torch
from torch.utils.data import Sampler, Dataset, DataLoader


class SinogramSampler(Sampler):
    """ Something that should behave like slice(start, end, step) to work as vectorized code
        on sinograms that are possibly too large to fit in memory.  """
    # todo: not used yet, but check https://stackoverflow.com/questions/58834338/how-does-the-getitem-s-idx-work-within-pytorchs-dataloader
    def __init__(self, projection_indices, disabled_projections = None, **kwargs):
        """
        Args:
            projection_indices: full list/range of projection indices available for sampling.
            disabled_projections: projection indices to exclude from sampling (e.g. known bad
                projections), removed before subsets are built.
        """
        super().__init__()
        self.enabled_projections = self.projection_indices = projection_indices
        self.disable_projections(disabled_projections)
        self._init_subsets()
        self.subset_index = 0
        self.reshuffle = False

    def _init_subsets(self):
        """Build `self.subsets`, the list of projection-index groups handed out one at a
        time by `next_projections`. Base implementation puts everything in a single subset;
        subclasses override this to split `enabled_projections` differently."""
        self.subsets = [self.enabled_projections]

    @property
    def num_projections(self):
        return len(self.enabled_projections)

    def disable_projections(self, disabled_projections):
        """Remove `disabled_projections` from `enabled_projections` (recomputed from the
        original `projection_indices` each time, so this is not cumulative across calls)."""
        removal_set = set(disabled_projections) if disabled_projections is not None else set([])
        self.enabled_projections = [n for n in self.projection_indices if n not in removal_set]

    @property
    def num_subsets(self):
        return len(self.subsets)

    @property
    def subset_index(self):
        return self._subset_index

    @property
    def subset_size(self):
        return len(self.subsets[0])

    @subset_index.setter
    def subset_index(self, index):
        """Advance to `index`, wrapping around to 0 once all subsets have been used. If
        `self.reshuffle` is set, wrapping also rebuilds the subsets via `_init_subsets`
        (e.g. reshuffling `OrderedSubsetSampler` between full sweeps)."""
        if index >= len(self.subsets):
            index = index % len(self.subsets)
            if self.reshuffle:
                self._init_subsets()
        self._subset_index = index

    def next_projections(self):
        """Return the current subset's projection indices as a tensor, then advance
        `subset_index` to the next subset (wrapping/reshuffling as needed)."""
        sampled_projections = self.subsets[self.subset_index]
        sampled_projections = torch.tensor(sampled_projections, dtype = torch.int32)
        self.subset_index = self.subset_index + 1
        return sampled_projections


    def get_os_epoch_dataloader(self):
        """Build a `DataLoader` whose items are meaningless placeholders: its only purpose
        is to make Lightning call `training_step` once per subset (`num_subsets` items ->
        one epoch = one full OS sweep), since the actual subset data lives in this sampler,
        not in the dataloader."""
        num_projections = self.num_projections
        # Convenience method to create a dataloader where one epoch is one OS sweep.
        class OrderedSubsetEpochDataset(Dataset):
            """
            Dataset that defines ONE epoch as ONE full OS sweep.
            Each __getitem__ corresponds to ONE OS subset update.
            """

            def __init__(self, num_subsets):
                self.num_subsets = num_subsets

            def __len__(self):
                # One epoch = one sweep over all subsets
                return self.num_subsets

            def __getitem__(self, idx):
                # No data needed — OS sampler lives in the model
                # Don't return None or the training_step won't be called
                # Return a
                return torch.arange(0, num_projections)

        os_loader = DataLoader(OrderedSubsetEpochDataset(self.num_subsets), batch_size=None, shuffle=False, num_workers=0)
        return os_loader


class StrideSampler(SinogramSampler):
    """Splits projections into `step` subsets by index modulo `step` (interleaved strides).

    NOTE: not currently used/instantiated anywhere in src/ or scripts/ - `OrderedSubsetSampler`
    is the sampler actually used by the reconstruction scripts.
    """
    def __init__(self, projection_indices, step, **kwargs):
        self.step = step
        super().__init__(projection_indices, **kwargs)

    def _init_subsets(self):
        self.subsets = [
            [n for n in self.enabled_projections if n % self.step == i]
            for i in range(self.step)]

    def __len__(self):
        return self.num_projections // self.step


class SequentialSampler(SinogramSampler):
    """Splits projections into consecutive, non-overlapping chunks of `batch_size` (in the
    order they appear in `enabled_projections`), unlike `OrderedSubsetSampler` which shuffles
    first.

    NOTE: not currently used/instantiated anywhere in src/ or scripts/ - `OrderedSubsetSampler`
    is the sampler actually used by the reconstruction scripts.
    """
    def __init__(self, projection_indices, batch_size, **kwargs):
        self.batch_size = batch_size
        super().__init__(projection_indices, **kwargs)

    def _init_subsets(self):
        self.subsets = [
            self.enabled_projections[i:i + self.batch_size]
            for i in range(0, len(self.enabled_projections), self.batch_size)
        ]


class OrderedSubsetSampler(SinogramSampler):
    """Samples a random subset of indices, internally sorted within subset"""
    def __init__(self, projection_indices, subset_size, **kwargs):
        self.batch_size = subset_size
        super().__init__(projection_indices, **kwargs)

    def _init_subsets(self):
        """Shuffle all enabled projections, then chop them into consecutive (internally
        sorted) chunks of `batch_size` - the classic ordered-subset split."""
        indices = list(self.enabled_projections)
        random.shuffle(indices)  # Random shuffle
        self.subsets = [sorted(indices[i:i + self.batch_size])
                        for i in range(0, len(indices), self.batch_size)]

    def __len__(self):
        return self.num_projections // self.batch_size


class SingleStepDataset(torch.utils.data.Dataset):
    """Trivial one-item dataset used to drive Lightning's validation loop: since validation
    in `CTReconstruction` doesn't consume per-item data (it uses its own fixed
    `validation_projections`), this just makes `validation_step` run exactly once per
    validation pass."""
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(0)
