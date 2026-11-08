"""Quality-control filters for MaxQuant evidence and GlyGly(K)Sites DataFrames.

Public interface
----------------
filter_evidence(df_evidence) -> pd.DataFrame
filter_ksites(df_ksites)     -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Evidence filters
# ---------------------------------------------------------------------------

_MIN_SCORE = 40.0
_MIN_DELTA_SCORE = 20.0
_MIN_LENGTH = 7
_MAX_MASS_ERROR_PPM = 5.0  # applied to abs(ppm) — values can be negative
_MAX_PEP = 0.01

_CONTAMINANT_PREFIXES = ("REV_", "Cont_")


def filter_evidence(df_evidence: pd.DataFrame) -> pd.DataFrame:
    """Return rows of *df_evidence* that pass all QC thresholds.

    Filters applied (all must hold for a row to be retained):
    - Reverse != '+'
    - Potential contaminant != '+'
    - No entry in Proteins starts with REV_ or Cont_
    - Score >= 40
    - Delta score >= 20
    - Length >= 7
    - abs(Mass error [ppm]) <= 5
    - PEP < 0.01
    """
    mask = (
        (df_evidence["Reverse"] != "+")
        & (df_evidence["Potential contaminant"] != "+")
        & ~df_evidence["Proteins"].apply(
            lambda proteins: any(
                p.startswith(_CONTAMINANT_PREFIXES) for p in proteins
            )
        )
        & (df_evidence["Score"] >= _MIN_SCORE)
        & (df_evidence["Delta score"] >= _MIN_DELTA_SCORE)
        & (df_evidence["Length"] >= _MIN_LENGTH)
        & (df_evidence["Mass error [ppm]"].abs() <= _MAX_MASS_ERROR_PPM)
        & (df_evidence["PEP"] < _MAX_PEP)
    )
    return df_evidence.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# K-site filters
# ---------------------------------------------------------------------------

_MIN_LOCALIZATION_PROB = 0.75


def filter_ksites(df_ksites: pd.DataFrame) -> pd.DataFrame:
    """Return rows of *df_ksites* that pass all QC thresholds.

    Filters applied (all must hold for a row to be retained):
    - Protein does not start with REV_ or Cont_
    - Localization prob >= 0.75
    """
    mask = (
        ~df_ksites["Protein"].str.startswith(_CONTAMINANT_PREFIXES)
        & (df_ksites["Localization prob"] >= _MIN_LOCALIZATION_PROB)
    )
    return df_ksites.loc[mask].reset_index(drop=True)
