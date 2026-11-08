"""
Fetch UniProt amino-acid sequences and write them to a FASTA file.

Spec: project_overview/components/04_fasta_retriever.yaml
"""

import logging
import pathlib

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
REQUEST_TIMEOUT_SECONDS = 120  # allow ample time for the streaming response


def retrieve_fasta(protein_ids: list[str], out_path: pathlib.Path) -> pathlib.Path:
    """Fetch FASTA sequences for *protein_ids* from UniProt and write to *out_path*.

    Parameters
    ----------
    protein_ids:
        Unique UniProt accession IDs (e.g. ``["P05161", "P41226"]``).
    out_path:
        Destination path for the output FASTA file.

    Returns
    -------
    pathlib.Path
        Same as *out_path*; returned for chaining.
    """
    if not protein_ids:
        raise ValueError("protein_ids must be non-empty")

    unique_ids = list(dict.fromkeys(protein_ids))  # deduplicate, preserve order

    query = " OR ".join(f"accession:{acc}" for acc in unique_ids)

    with tqdm(total=len(unique_ids), desc="Fetching FASTA", unit="accession") as pbar:
        # Prefer the streaming endpoint; fall back to /search if unreachable.
        for url in (UNIPROT_STREAM_URL, UNIPROT_SEARCH_URL):
            try:
                response = requests.get(
                    url,
                    params={"query": query, "format": "fasta"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                break
            except requests.exceptions.ReadTimeout:
                if url == UNIPROT_SEARCH_URL:
                    raise  # both endpoints failed
                logger.warning(
                    "Stream endpoint timed out; retrying with /search endpoint"
                )
        pbar.update(len(unique_ids))

    fasta_text = response.text

    # Warn for any accession whose sequence has no lysine residue
    for block in fasta_text.split(">")[1:]:  # skip empty leading element
        lines = block.splitlines()
        sequence = "".join(lines[1:])
        if "K" not in sequence:
            header = lines[0]
            logger.warning("No lysine residue found in protein: %s", header)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fasta_text, encoding="utf-8")

    return out_path
