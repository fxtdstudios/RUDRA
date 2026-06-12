"""Highlight-balanced sampling utilities for HDR datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SampleHDRStats:
    index: int
    peak: float
    p95: float
    mean: float
    highlight_fraction: float


def bucket_from_stats(s: SampleHDRStats) -> str:
    if s.peak > 16.0 or s.highlight_fraction > 0.05:
        return "extreme_highlight"
    if s.peak > 4.0 or s.p95 > 1.5:
        return "high_dynamic_range"
    if s.peak > 1.2:
        return "medium_dynamic_range"
    return "low_dynamic_range"


class HighlightBalancedSampler(Sampler[int]):
    """Round-robin sampler across dynamic-range buckets.

    It prevents training from being dominated by ordinary low-DR images.
    """

    def __init__(self, stats: Sequence[SampleHDRStats], shuffle: bool = True, seed: int = 1234):
        self.stats = list(stats)
        self.shuffle = shuffle
        self.seed = seed
        buckets: dict[str, list[int]] = defaultdict(list)
        for s in self.stats:
            buckets[bucket_from_stats(s)].append(s.index)
        self.buckets = {k: v for k, v in buckets.items() if v}

    def __len__(self) -> int:
        return len(self.stats)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        buckets = {}
        for k, vals in self.buckets.items():
            vals = list(vals)
            if self.shuffle:
                perm = torch.randperm(len(vals), generator=g).tolist()
                vals = [vals[i] for i in perm]
            buckets[k] = vals
        keys = list(buckets.keys())
        ptr = {k: 0 for k in keys}
        yielded = 0
        while yielded < len(self.stats):
            progressed = False
            for k in keys:
                if ptr[k] < len(buckets[k]):
                    yield buckets[k][ptr[k]]
                    ptr[k] += 1
                    yielded += 1
                    progressed = True
                    if yielded >= len(self.stats):
                        break
            if not progressed:
                break
