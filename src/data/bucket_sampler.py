"""
Bucket sampler: groups sequences by length so padding waste is minimised.

Components:
  HDF5Dataset       -- loads embeddings from embeddings.h5 by split
  BucketSampler     -- yields same-bucket batches
  create_dataloader -- factory that wires dataset + sampler + collate_fn
"""

import pathlib
import random
from collections import defaultdict
from typing import Iterator

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class HDF5Dataset(Dataset):
    """Dataset backed by an HDF5 file produced by the embedder.

    HDF5 layout expected:
        /embeddings/<protein_id>  shape (L, 1024) float32
        /metadata                 pandas DataFrame with columns:
                                  [protein_id, split, target_k_index, label]

    Reads embeddings lazily in __getitem__ so DataLoader workers each open
    their own file handle (multiprocessing-safe).
    """

    def __init__(self, h5_path: pathlib.Path, split: str) -> None:
        self.h5_path = h5_path
        # Load and filter metadata at construction time (small, cheap)
        meta: pd.DataFrame = pd.read_hdf(str(h5_path), key="metadata")
        self._meta = meta[meta["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int):
        row = self._meta.iloc[idx]
        protein_id: str = row["protein_id"]
        target_k_index: int = int(row["target_k_index"])
        label: float = float(row["label"])

        # Open file per call — safe for multiprocessing workers
        with h5py.File(self.h5_path, "r") as f:
            embedding = torch.tensor(
                np.array(f["embeddings"][protein_id]), dtype=torch.float32
            )  # shape (L, 1024)

        return embedding, target_k_index, label


class BucketSampler(Sampler):
    """Yields batches whose samples all share the same length bucket.

    Bucket id = length // bucket_width, so bucket 0 covers [0, bucket_width),
    bucket 1 covers [bucket_width, 2*bucket_width), etc.

    Within each bucket samples are shuffled randomly before batching.
    """

    def __init__(
        self, lengths: list[int], batch_size: int, bucket_width: int
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_width = bucket_width

    # ------------------------------------------------------------------
    # Build bucket → index mapping (recomputed each epoch via __iter__)
    # ------------------------------------------------------------------

    def _make_batches(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            bucket_id = length // self.bucket_width
            buckets[bucket_id].append(idx)

        batches: list[list[int]] = []
        for indices in buckets.values():
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])

        random.shuffle(batches)  # shuffle bucket order across epochs
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._make_batches()

    def __len__(self) -> int:
        # Approximate: each sample contributes to exactly one batch
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size
            for group in _group_by_bucket(self.lengths, self.bucket_width).values()
        )


def _group_by_bucket(lengths: list[int], bucket_width: int) -> dict[int, list[int]]:
    """Helper used by BucketSampler.__len__ without constructing the object."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, length in enumerate(lengths):
        buckets[length // bucket_width].append(idx)
    return buckets


def _collate_fn(batch):
    """Pad embeddings to the longest sequence in this batch.

    Args:
        batch: list of (embedding, target_k_index, label) tuples

    Returns:
        x               (B, L_max, 1024) float32
        seq_lengths     (B,)             int64
        target_k_indices (B,)            int64
        labels          (B,)             float32
    """
    embeddings, target_k_indices, labels = zip(*batch)

    seq_lengths = torch.tensor([e.shape[0] for e in embeddings], dtype=torch.int64)
    L_max = int(seq_lengths.max().item())

    B = len(embeddings)
    x = torch.zeros(B, L_max, 1024, dtype=torch.float32)
    for i, emb in enumerate(embeddings):
        L = emb.shape[0]
        x[i, :L, :] = emb

    return (
        x,
        seq_lengths,
        torch.tensor(target_k_indices, dtype=torch.int64),
        torch.tensor(labels, dtype=torch.float32),
    )


def create_dataloader(
    h5_path: pathlib.Path,
    split: str,
    batch_size: int,
    bucket_width: int,
) -> DataLoader:
    """Create a DataLoader grouped by sequence-length buckets.

    Returns batches of (x, seq_lengths, target_k_indices, labels).
    """
    dataset = HDF5Dataset(h5_path, split)

    # Gather lengths: load each embedding's shape[0] from the HDF5 file.
    # Done once at construction; acceptable cost (shape read, not data read).
    with h5py.File(h5_path, "r") as f:
        lengths = [
            f["embeddings"][dataset._meta.iloc[i]["protein_id"]].shape[0]
            for i in range(len(dataset))
        ]

    sampler = BucketSampler(lengths, batch_size, bucket_width)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_collate_fn,
    )
