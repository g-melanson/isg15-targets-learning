"""Embed protein sequences with ProtT5 and write to a compressed HDF5 registry.

Public interface
----------------
embed_sequences(fasta_path, split_assignments, df_corpus, out_path) -> None
_write_embeddings_to_hdf5(protein_id, embedding, h5_path) -> None  # testable helper
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.lysine import parse_fasta

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_sequences(
    fasta_path: pathlib.Path,
    split_assignments: pd.DataFrame,   # schema: {protein_id: str, split: str}
    df_corpus: pd.DataFrame,           # schema: {Protein, target_k_index, label, source_dataset, sequence}
    out_path: pathlib.Path,            # e.g. data/embeddings.h5
    device: torch.device = torch.device("cpu"),
) -> None:
    """Embed each protein in split_assignments with ProtT5 and write to HDF5.

    The HDF5 file is built incrementally:
    - ``/embeddings/<protein_id>``  shape (L, 1024) float32, gzip-compressed
    - ``/metadata``                 pandas table: protein_id, split, target_k_index, label

    Re-running is safe: proteins already present in ``/embeddings/`` are skipped
    (idempotent).

    Parameters
    ----------
    fasta_path:
        Path to a UniProt FASTA file containing the sequences to embed.
    split_assignments:
        DataFrame with columns ``protein_id`` and ``split``.
    df_corpus:
        DataFrame with columns ``Protein``, ``target_k_index``, ``label``,
        ``source_dataset``, ``sequence``.
    out_path:
        Destination HDF5 path.
    device:
        Torch device for ProtT5 inference. Defaults to CPU; pass
        ``torch.device("cuda")`` on a GPU machine to use the GPU.
    """
    # Import ProtT5 here — not at module level — so the module is importable
    # even when the model weights have not been downloaded.
    from transformers import T5EncoderModel, T5Tokenizer  # noqa: PLC0415

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    sequences: dict[str, str] = parse_fasta(fasta_path)

    protein_ids = split_assignments["protein_id"].tolist()

    for protein_id in tqdm(protein_ids, desc="Embedding", unit="protein"):
        # --- Idempotency check: skip proteins already in the registry --------
        with h5py.File(out_path, "a") as h5:
            if protein_id in h5.get("embeddings", {}):
                continue  # already embedded — skip

        seq = sequences[protein_id]
        # ProtT5 convention: space-separated residues
        spaced_seq = " ".join(list(seq))

        ids = tokenizer(spaced_seq, return_tensors="pt", add_special_tokens=True)
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            output = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])

        # output.last_hidden_state: (1, L+1, 1024) — drop batch dim and EOS token
        embedding: np.ndarray = (
            output.last_hidden_state[0, : len(seq), :]
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # shape (L, 1024)

        _write_embeddings_to_hdf5(protein_id, embedding, out_path)

        # Update tqdm postfix with current protein for observability
        tqdm.write(f"  embedded {protein_id}  shape={embedding.shape}")

    # --- Write /metadata table -----------------------------------------------
    df_meta = (
        df_corpus[df_corpus["Protein"].isin(split_assignments["protein_id"])]
        .merge(
            split_assignments,
            left_on="Protein",
            right_on="protein_id",
            how="inner",
        )[["protein_id", "split", "target_k_index", "label"]]
        .reset_index(drop=True)
    )
    df_meta.to_hdf(out_path, key="metadata", mode="a", format="table")


# ---------------------------------------------------------------------------
# Testable helper (also used internally)
# ---------------------------------------------------------------------------


def _write_embeddings_to_hdf5(
    protein_id: str,
    embedding: np.ndarray,  # (L, 1024) float32
    h5_path: pathlib.Path,
) -> None:
    """Write a single protein embedding to the HDF5 registry.

    Idempotent: if the dataset already exists it is left unchanged.

    Parameters
    ----------
    protein_id:
        UniProt accession used as the dataset key.
    embedding:
        NumPy array of shape ``(L, 1024)`` and dtype float32.
    h5_path:
        Path to the HDF5 file (created if absent).
    """
    with h5py.File(h5_path, "a") as h5:
        grp = h5.require_group("embeddings")
        if protein_id not in grp:
            grp.create_dataset(protein_id, data=embedding, compression="gzip")
