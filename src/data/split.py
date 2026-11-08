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
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner

logger = logging.getLogger(__name__)

# --- constants (one authoritative place) ---
IDENTITY_THRESHOLD = 0.30
TEST_RATIO = 0.15
VAL_RATIO = 0.15

# graph-part cluster IDs after train_val_test_split (see train_val_test_split.py)
# cluster 0 → train, 1 → test, 2 → val
_CLUSTER_TO_SPLIT: Dict[int, str] = {0: "train", 1: "test", 2: "val"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_accession(record_id: str) -> str:
    """Return UniProt accession from a FASTA record ID.

    UniProt format: ``sp|ACCESSION|NAME`` or ``tr|ACCESSION|NAME``.
    Falls back to the full ID string for non-UniProt headers.
    """
    parts = record_id.split("|")
    if len(parts) >= 2 and parts[0] in ("sp", "tr"):
        return parts[1]
    return record_id


def _write_graphpart_fasta(
    accession_to_seq: Dict[str, str], fasta_path: pathlib.Path
) -> None:
    """Write a simplified FASTA where each header is the bare accession.

    graph-part parses the first pipe-delimited field as the identifier, so
    standard UniProt ``>sp|ACC|NAME`` headers would all resolve to ``"sp"``.
    Writing ``>ACC`` avoids that collision.
    """
    with open(fasta_path, "w") as f:
        for accession, seq in accession_to_seq.items():
            f.write(f">{accession}\n{seq}\n")


def _compute_pairwise_identities(
    accession_to_seq: Dict[str, str],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    """Return (id1, id2, identity) for all pairs with identity > threshold.

    Identity = max(0, score) / min(len(seq1), len(seq2)), using global
    alignment with match=1, mismatch=0, and moderate gap penalties.

    Why moderate gap penalties (not 0)?  With gap_open=0 the aligner can
    freely rearrange both sequences, turning identity into an LCS metric;
    random 100-aa sequences score ~33 % that way.  A gap_open=-1 penalty
    prevents free insertions and gives realistic ~5-15 % identity for
    unrelated proteins.

    Complexity: O(N² × L²) where L is mean sequence length.
    Fine for small-to-medium datasets; for >1 000 proteins consider
    switching to graphpart's mmseqs2 mode once MMseqs2 is available.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1   # prevents free gap insertions
    aligner.extend_gap_score = -0.1

    ids = list(accession_to_seq.keys())
    edges: List[Tuple[str, str, float]] = []

    for i, j in combinations(range(len(ids)), 2):
        id1, id2 = ids[i], ids[j]
        seq1, seq2 = accession_to_seq[id1], accession_to_seq[id2]
        raw_score = aligner.score(seq1, seq2)
        identity = max(0.0, raw_score) / min(len(seq1), len(seq2))
        if identity > threshold:
            edges.append((id1, id2, identity))

    logger.info(
        "Pairwise identity: %d / %d pairs above %.0f%% threshold",
        len(edges),
        len(ids) * (len(ids) - 1) // 2,
        threshold * 100,
    )
    return edges


def _write_edge_file(
    edges: List[Tuple[str, str, float]], edge_path: pathlib.Path
) -> None:
    """Write edge CSV for graph-part's precomputed mode (no header).

    Format per line: ``query_id,library_id,identity``
    """
    with open(edge_path, "w") as f:
        for id1, id2, identity in edges:
            f.write(f"{id1},{id2},{identity:.6f}\n")


def _run_graphpart(
    fasta_path: pathlib.Path,
    edge_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Call the graphpart CLI in precomputed mode.

    Uses ``one-minus`` transformation so the identity edge values are
    converted to distances internally: pairs with identity > 0.30 become
    distance < 0.70 and are kept connected (same partition).

    Raises RuntimeError if graphpart exits non-zero.
    """
    graphpart_bin = pathlib.Path(sys.executable).parent / "graphpart"
    cmd = [
        str(graphpart_bin),
        "precomputed",
        "-ff", str(fasta_path),
        "-ef", str(edge_path),
        "-th", str(IDENTITY_THRESHOLD),
        "-tf", "one-minus",
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
    """Parse graphpart's output CSV into the required protein_id / split schema.

    graphpart writes: index=AC, columns include ``cluster`` (float: 0=train,
    1=test, 2=val after train_val_test_split merging).
    """
    raw = pd.read_csv(graphpart_csv, index_col=0)
    df = pd.DataFrame(
        {
            "protein_id": raw.index.astype(str),
            "split": raw["cluster"].astype(int).map(_CLUSTER_TO_SPLIT),
        }
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def split_by_homology(
    fasta_path: pathlib.Path, out_path: pathlib.Path
) -> pd.DataFrame:
    """
    Use graph-part to assign proteins to train/val/test splits with
    no pair sharing > 30% sequence identity across split boundaries.

    Args:
        fasta_path: Path to proteins.fasta (from fasta_retriever)
        out_path: Destination CSV path (e.g. data/split_assignments.csv)

    Returns:
        pd.DataFrame with columns: protein_id (str), split (str in {"train","val","test"})
    """
    # 1. Parse FASTA; extract bare accessions for graph-part compatibility
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

        # 2. Write simplified FASTA (bare accession headers)
        _write_graphpart_fasta(accession_to_seq, simplified_fasta)

        # 3. Compute pairwise identities; write edge file (may be empty)
        edges = _compute_pairwise_identities(accession_to_seq, IDENTITY_THRESHOLD)
        _write_edge_file(edges, edge_file)

        # 4. Run graph-part (precomputed mode — no external alignment tool needed)
        _run_graphpart(simplified_fasta, edge_file, graphpart_out)

        # 5. Parse graph-part output
        df = _parse_graphpart_csv(graphpart_out)

    # 6. Write standardised CSV to caller-specified path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print counts per split (spec requirement)
    counts = df["split"].value_counts()
    print("Split assignments (protein counts):")
    print(counts.to_string())

    return df
