"""Build a lysine-site index from labelled protein sets and sequence data.

Public interface
----------------
parse_fasta(fasta_path) -> dict[str, str]
build_lysine_index(df_modified, df_unmodified, sequences) -> pd.DataFrame
"""

from __future__ import annotations

import pathlib

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_fasta(fasta_path: pathlib.Path) -> dict[str, str]:
    """Parse a UniProt FASTA file into an accession → sequence mapping.

    Handles both reviewed (sp|) and unreviewed (tr|) UniProt headers of the
    form ``>sp|P05161|...`` or ``>tr|P05161|...``.

    Parameters
    ----------
    fasta_path:
        Path to a FASTA file.

    Returns
    -------
    dict[str, str]
        ``{accession: sequence}`` where accession is the UniProt ID.
    """
    sequences: dict[str, str] = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        # UniProt header format: sp|ACCESSION|ENTRY_NAME or tr|ACCESSION|...
        parts = record.id.split("|")
        accession = parts[1] if len(parts) >= 2 else record.id
        sequences[accession] = str(record.seq)
    return sequences


def build_lysine_index(
    df_modified: pd.DataFrame,
    df_unmodified: pd.DataFrame,
    sequences: dict[str, str],
) -> pd.DataFrame:
    """Assign a 0-indexed target_k_index to every lysine site in the corpus.

    Three passes produce positive (label=1) and negative (label=0) rows:

    Pass 1 — detected ISGylation sites from df_modified (label=1).
    Pass 2 — undetected lysines in modified proteins (label=0).
    Pass 3 — all lysines in unmodified-only proteins (label=0).

    Parameters
    ----------
    df_modified:
        Output of build_corpus().
        Schema: source_dataset, site_id, Protein, Position, Localization prob, label.
    df_unmodified:
        Output of build_corpus().
        Schema: source_dataset, Proteins, label.
    sequences:
        Mapping of UniProt accession → full amino-acid sequence.

    Returns
    -------
    pd.DataFrame
        Columns: Protein, target_k_index, label, source_dataset, sequence.
    """
    rows_pass1 = _pass1_positive_sites(df_modified, sequences)
    rows_pass2 = _pass2_undetected_lysines(df_modified, sequences, rows_pass1)
    rows_pass3 = _pass3_unmodified_lysines(df_unmodified, df_modified, sequences)

    df_corpus = pd.concat(
        [rows_pass1, rows_pass2, rows_pass3],
        ignore_index=True,
    )

    # Enforce output dtypes
    df_corpus["target_k_index"] = df_corpus["target_k_index"].astype(int)
    df_corpus["label"] = df_corpus["label"].astype(int)

    return df_corpus.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pass1_positive_sites(
    df_modified: pd.DataFrame,
    sequences: dict[str, str],
) -> pd.DataFrame:
    """Pass 1: one positive row per df_modified row."""
    records = []
    for _, row in tqdm(df_modified.iterrows(), total=len(df_modified), desc="Pass 1 — positive k-sites"):
        protein = row["Protein"]
        # Skip rows whose protein has no sequence (e.g. empty accession from dirty join)
        if protein not in sequences:
            continue
        seq = sequences[protein]
        position = int(row["Position"])
        raw_index = position - 1  # MaxQuant is 1-indexed → 0-indexed
        # Normalize any negative index (Position=0 produces -1; Python wraps it to seq[-1])
        target_k_index = raw_index if raw_index >= 0 else len(seq) + raw_index
        # Guard: skip if index is still out-of-range or doesn't land on a 'K'
        if target_k_index < 0 or target_k_index >= len(seq) or seq[target_k_index] != "K":
            continue
        records.append({
            "Protein": protein,
            "target_k_index": target_k_index,
            "label": 1,
            "source_dataset": row["source_dataset"],
            "sequence": seq,
        })
    return pd.DataFrame(records)


def _pass2_undetected_lysines(
    df_modified: pd.DataFrame,
    sequences: dict[str, str],
    df_pass1: pd.DataFrame,
) -> pd.DataFrame:
    """Pass 2: negative rows for every undetected K in modified proteins."""
    # Build a set of (protein, k_index) pairs that are already in Pass 1
    detected: set[tuple[str, int]] = set(
        zip(df_pass1["Protein"], df_pass1["target_k_index"])
    )

    # Protein → its first source_dataset occurrence
    source_by_protein: dict[str, str] = (
        df_modified.groupby("Protein")["source_dataset"].first().to_dict()
    )

    records = []
    for protein, source_dataset in tqdm(
        source_by_protein.items(),
        desc="Pass 2 — undetected lysines in modified proteins",
    ):
        seq = sequences[protein]
        for k_index, aa in enumerate(seq):
            if aa == "K" and (protein, k_index) not in detected:
                records.append({
                    "Protein": protein,
                    "target_k_index": k_index,
                    "label": 0,
                    "source_dataset": source_dataset,
                    "sequence": seq,
                })
    return pd.DataFrame(records)


def _pass3_unmodified_lysines(
    df_unmodified: pd.DataFrame,
    df_modified: pd.DataFrame,
    sequences: dict[str, str],
) -> pd.DataFrame:
    """Pass 3: negative rows for every K in unmodified-only proteins."""
    modified_proteins: set[str] = set(df_modified["Protein"])

    # Explode the Proteins list column; keep proteins not in modified set
    df_exploded = (
        df_unmodified[["source_dataset", "Proteins"]]
        .explode("Proteins")
        .rename(columns={"Proteins": "Protein"})
        .drop_duplicates(subset=["Protein"])
        .reset_index(drop=True)
    )
    df_exploded = df_exploded[~df_exploded["Protein"].isin(modified_proteins)]

    records = []
    for _, row in tqdm(
        df_exploded.iterrows(),
        total=len(df_exploded),
        desc="Pass 3 — lysines in unmodified proteins",
    ):
        protein = row["Protein"]
        if protein not in sequences:
            continue
        seq = sequences[protein]
        for k_index, aa in enumerate(seq):
            if aa == "K":
                records.append({
                    "Protein": protein,
                    "target_k_index": k_index,
                    "label": 0,
                    "source_dataset": row["source_dataset"],
                    "sequence": seq,
                })
    return pd.DataFrame(records)
