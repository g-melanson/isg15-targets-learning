"""Join filtered evidence PSMs to k-site records and produce labelled sets.

Public interface
----------------
build_corpus(df_evidence_clean, df_ksites_clean) -> (df_modified, df_unmodified)
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_corpus(
    df_evidence_clean: pd.DataFrame,
    df_ksites_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce labelled modified and unmodified PSM sets.

    Parameters
    ----------
    df_evidence_clean:
        Output of filter_evidence().  Must contain columns:
        source_dataset, Proteins, GlyGly (K), GlyGly (K) site IDs.
    df_ksites_clean:
        Output of filter_ksites().  Must contain columns:
        source_dataset, id, Protein, Position, Localization prob.

    Returns
    -------
    df_modified : pd.DataFrame
        One row per successfully matched PSM × k-site pair.
        Columns: source_dataset, site_id, Protein, Position,
                 Localization prob, label (=1).
    df_unmodified : pd.DataFrame
        One row per unmodified PSM whose proteins are absent
        from the modified protein set.
        Columns: source_dataset, Proteins, label (=0).
    """
    df_modified = _build_modified(df_evidence_clean, df_ksites_clean)
    df_unmodified = _build_unmodified(df_evidence_clean, df_modified)

    _print_summary(df_modified, df_unmodified)
    return df_modified, df_unmodified


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_modified(
    df_evidence_clean: pd.DataFrame,
    df_ksites_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Build df_modified (label=1).

    Steps:
      1. Keep evidence rows where GlyGly (K) > 0.
      2. Explode GlyGly (K) site IDs → one row per site_id.
      3. Inner-join to df_ksites_clean on (source_dataset, site_id=id).
         Join is not perfectly lossless: some site IDs reference k-sites
         that were removed by the localization-prob filter.  Lost rows are
         logged but not treated as errors because the data loss is expected.
      4. Add label=1 and select final columns.
    """
    # Step 1: modified PSMs only
    df_glygly = df_evidence_clean[df_evidence_clean["GlyGly (K)"] > 0].copy()

    # Step 2: explode site IDs → one row per site_id
    df_exploded = (
        df_glygly
        .explode("GlyGly (K) site IDs")
        .rename(columns={"GlyGly (K) site IDs": "site_id"})
        .reset_index(drop=True)
    )

    # Ensure site_id is int (explode from list[int] may leave object dtype)
    df_exploded["site_id"] = df_exploded["site_id"].astype(int)

    n_exploded = len(df_exploded)

    # Step 3: inner join — keeps only rows with a matching k-site record
    df_joined = df_exploded.merge(
        df_ksites_clean[["source_dataset", "id", "Protein", "Position", "Localization prob"]],
        left_on=["source_dataset", "site_id"],
        right_on=["source_dataset", "id"],
        how="inner",
    ).reset_index(drop=True)

    n_joined = len(df_joined)
    n_lost = n_exploded - n_joined
    match_rate = n_joined / n_exploded if n_exploded > 0 else 0.0

    if n_lost > 0:
        print(
            f"[build_corpus] Join loss: {n_lost} of {n_exploded} exploded rows "
            f"({n_lost / n_exploded * 100:.1f}%) had no matching k-site record. "
            f"This is expected when k-sites were removed by the localization-prob filter."
        )

    print(f"[build_corpus] Join match-rate: {match_rate:.3f} ({n_joined}/{n_exploded})")

    # Step 4: label and select columns
    df_joined["label"] = 1

    df_modified = df_joined[
        ["source_dataset", "site_id", "Protein", "Position", "Localization prob", "label"]
    ].reset_index(drop=True)

    # Enforce output dtypes
    df_modified["site_id"] = df_modified["site_id"].astype(int)
    df_modified["Position"] = df_modified["Position"].astype(int)
    df_modified["Localization prob"] = df_modified["Localization prob"].astype(float)
    df_modified["label"] = df_modified["label"].astype(int)

    return df_modified


def _build_unmodified(
    df_evidence_clean: pd.DataFrame,
    df_modified: pd.DataFrame,
) -> pd.DataFrame:
    """Build df_unmodified (label=0).

    Steps:
      1. Keep evidence rows where GlyGly (K) == 0.
      2. Build the modified protein set from df_modified["Protein"].
      3. Keep only rows where NONE of the row's Proteins appear in
         the modified protein set.
      4. Add label=0 and select final columns.
    """
    df_unmod_candidates = df_evidence_clean[df_evidence_clean["GlyGly (K)"] == 0].copy()

    modified_proteins: set[str] = set(df_modified["Protein"])

    # Row is safe if no protein in its list overlaps with modified set
    no_overlap_mask = df_unmod_candidates["Proteins"].apply(
        lambda proteins: not any(p in modified_proteins for p in proteins)
    )

    df_unmod = df_unmod_candidates.loc[no_overlap_mask].copy().reset_index(drop=True)

    df_unmod["label"] = 0

    df_unmodified = df_unmod[["source_dataset", "Proteins", "label"]].reset_index(drop=True)
    df_unmodified["label"] = df_unmodified["label"].astype(int)

    return df_unmodified


def _print_summary(
    df_modified: pd.DataFrame,
    df_unmodified: pd.DataFrame,
) -> None:
    print(f"[build_corpus] df_modified rows:   {len(df_modified):>8,}")
    print(f"[build_corpus] df_unmodified rows: {len(df_unmodified):>8,}")
