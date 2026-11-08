"""Load MaxQuant output files into typed, NaN-free DataFrames.

Public interface
----------------
load_raw_data(dataset_dirs) -> (df_evidence, df_ksites)
"""

from __future__ import annotations

import pathlib
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Column selections — only these are read from disk; all others are dropped.
# ---------------------------------------------------------------------------

_EVIDENCE_COLS = [
    "Proteins",
    "Score",
    "Delta score",
    "Mass error [ppm]",
    "Length",
    "PEP",
    "Potential contaminant",
    "Reverse",
    "GlyGly (K) site IDs",
    "GlyGly (K)",
]

_KSITE_COLS = [
    "Protein",
    "Position",
    "Localization prob",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_str_list(series: pd.Series) -> pd.Series:
    """Split semicolon-delimited strings into lists of stripped tokens.

    NaN / empty string → empty list.
    """
    def _split(val: object) -> list:
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [t.strip() for t in str(val).split(";") if t.strip() != ""]

    return series.apply(_split)


def _split_int_list(series: pd.Series) -> pd.Series:
    """Split semicolon-delimited strings into lists of ints.

    NaN / empty string → empty list.
    """
    def _split(val: object) -> list:
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [int(t.strip()) for t in str(val).split(";") if t.strip() != ""]

    return series.apply(_split)


def _dataset_id(directory: pathlib.Path) -> str:
    """Derive the dataset ID from directory name, e.g. '8604_txt' → '8604'."""
    name = directory.name
    if name.endswith("_txt"):
        return name[: -len("_txt")]
    return name


# ---------------------------------------------------------------------------
# Per-file loaders
# ---------------------------------------------------------------------------


def _load_evidence(path: pathlib.Path, dataset_id: str) -> pd.DataFrame:
    """Load one evidence.txt file into a typed DataFrame."""
    raw = pd.read_csv(
        path,
        sep="\t",
        usecols=_EVIDENCE_COLS,
        low_memory=False,
    )

    # --- string columns: fill NaN with "" before any other processing ---
    raw["Potential contaminant"] = raw["Potential contaminant"].fillna("").astype(str)
    raw["Reverse"] = raw["Reverse"].fillna("").astype(str)

    # --- list columns: split from raw strings (NaN handled inside helpers) ---
    proteins = _split_str_list(raw["Proteins"])
    site_ids = _split_int_list(raw["GlyGly (K) site IDs"])

    # --- numeric columns: fill NaN with 0 then cast ---
    score = raw["Score"].fillna(0.0).astype(float)
    delta_score = raw["Delta score"].fillna(0.0).astype(float)
    mass_error = raw["Mass error [ppm]"].fillna(0.0).astype(float)
    length = raw["Length"].fillna(0).astype(int)
    pep = raw["PEP"].fillna(0.0).astype(float)
    glygly_k = raw["GlyGly (K)"].fillna(0).astype(int)

    df = pd.DataFrame(
        {
            "id": range(len(raw)),
            "Proteins": proteins,
            "Score": score,
            "Delta score": delta_score,
            "Mass error [ppm]": mass_error,
            "Length": length,
            "PEP": pep,
            "Potential contaminant": raw["Potential contaminant"],
            "Reverse": raw["Reverse"],
            "GlyGly (K) site IDs": site_ids,
            "GlyGly (K)": glygly_k,
            "source_dataset": dataset_id,
        }
    )

    df["id"] = df["id"].astype(int)
    df["source_dataset"] = df["source_dataset"].astype(str)
    return df


def _load_ksites(path: pathlib.Path, dataset_id: str) -> pd.DataFrame:
    """Load one GlyGly(K)Sites.txt file into a typed DataFrame."""
    raw = pd.read_csv(
        path,
        sep="\t",
        usecols=_KSITE_COLS,
        low_memory=False,
    )

    df = pd.DataFrame(
        {
            "id": range(len(raw)),
            "Protein": raw["Protein"].fillna("").astype(str),
            "Position": raw["Position"].fillna(0).astype(int),
            "Localization prob": raw["Localization prob"].fillna(0.0).astype(float),
            "source_dataset": dataset_id,
        }
    )

    df["id"] = df["id"].astype(int)
    df["source_dataset"] = df["source_dataset"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw_data(
    dataset_dirs: List[pathlib.Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MaxQuant evidence and GlyGly(K)Sites files from *dataset_dirs*.

    Each directory must contain:
      - evidence.txt
      - GlyGly (K)Sites.txt

    Parameters
    ----------
    dataset_dirs:
        Paths to MaxQuant output directories.  Each must contain the two
        required files.  The dataset ID is derived from the directory name
        (e.g. ``8604_txt`` → ``"8604"``).

    Returns
    -------
    df_evidence : pd.DataFrame
        Concatenated, typed evidence rows across all directories.
    df_ksites : pd.DataFrame
        Concatenated, typed GlyGly(K)Sites rows across all directories.
    """
    if not dataset_dirs:
        raise ValueError("dataset_dirs must not be empty")

    evidence_frames: List[pd.DataFrame] = []
    ksite_frames: List[pd.DataFrame] = []

    for directory in dataset_dirs:
        directory = pathlib.Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

        dataset_id = _dataset_id(directory)

        evidence_path = directory / "evidence.txt"
        ksite_path = directory / "GlyGly (K)Sites.txt"

        for required in (evidence_path, ksite_path):
            if not required.exists():
                raise FileNotFoundError(f"Required file missing: {required}")

        evidence_frames.append(_load_evidence(evidence_path, dataset_id))
        ksite_frames.append(_load_ksites(ksite_path, dataset_id))

    df_evidence = pd.concat(evidence_frames, ignore_index=True)
    df_ksites = pd.concat(ksite_frames, ignore_index=True)

    return df_evidence, df_ksites
