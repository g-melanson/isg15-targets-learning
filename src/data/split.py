"""
Homology-aware train/val/test dataset splitter using graph-part.

Partitions proteins into train/val/test so that no pair sharing
> 30% sequence identity appears across split boundaries.
"""

import logging
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import pandas as pd
from Bio import SeqIO

logger = logging.getLogger(__name__)

IDENTITY_THRESHOLD = 0.30
TEST_RATIO = 0.15
VAL_RATIO = 0.15

# if this becomes an enum
# then reuseable for training
_CLUSTER_TO_SPLIT: Dict[int, str] = {0: "train", 1: "test", 2: "val"}


def _extract_accession(record_id: str) -> str:
    parts = record_id.split("|")
    if len(parts) >= 2 and parts[0] in ("sp", "tr"):
        return parts[1]
    return record_id


def _write_graphpart_fasta(
    accession_to_seq: Dict[str, str], fasta_path: pathlib.Path
) -> None:
    with open(fasta_path, "w") as f:
        for accession, seq in accession_to_seq.items():
            f.write(f">{accession}\n{seq}\n")


def _run_graphpart(
    fasta_path: pathlib.Path,
    edge_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:

    graphpart_bin = pathlib.Path(sys.executable).parent / "graphpart"
    cmd = [
        str(graphpart_bin),
        "mmseqs2",
        "-ff", str(fasta_path),
        "-th", str(IDENTITY_THRESHOLD),
        "-te", str(TEST_RATIO),
        "-va", str(VAL_RATIO),
        "-of", str(out_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"graphpart failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

def _parse_graphpart_csv(graphpart_csv: pathlib.Path) -> pd.DataFrame:
    raw = pd.read_csv(graphpart_csv, index_col=0)
    return pd.DataFrame(
        {
            "protein_id": raw.index.astype(str),
            "split": raw["cluster"].astype(int).map(_CLUSTER_TO_SPLIT),
        }
    ).reset_index(drop=True)


def split_by_homology(
    fasta_path: pathlib.Path, out_path: pathlib.Path
) -> pd.DataFrame:

    # Parse FASTA 
    # extract bare accessions
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    accession_to_seq: Dict[str, str] = {
        _extract_accession(r.id): str(r.seq) for r in records
    }
    logger.info("Loaded %d proteins from %s", len(accession_to_seq), fasta_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        simplified_fasta = tmp / "proteins.fasta"
        edge_file = tmp / "edges.csv"
        graphpart_out = tmp / "graphpart_out.csv"

        _write_graphpart_fasta(accession_to_seq, simplified_fasta)

        _run_graphpart(simplified_fasta, edge_file, graphpart_out)

        df = _parse_graphpart_csv(graphpart_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    counts = df["split"].value_counts()
    print("Split assignments (protein counts):")
    print(counts.to_string())

    return df
