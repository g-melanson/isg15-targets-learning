# ISG15 Target-Site Prediction

Binary classifier that predicts whether a **lysine residue** in a human protein will be ISGylated (covalently modified by ISG15). Built on top of ProtT5 embeddings and a BiLSTM.

---

## Problem framing

ISGylation is a post-translational modification (PTM) where ISG15 is conjugated to a substrate lysine. This process occurs generally through co-translation as protein targets are made, or through selective pathways enabling complex system cross-talking. Current analyses have yet to provide a strong opinion **how** proteins are selected for modification given the apparent selectivity of the targets across different experimental models. 

Finding testable insights is difficult because lysine modification is shared across multiple ubiquitin-like families, and therefore likely depend on complex factors like protein topology, surface features, and biological background. Whereas typical approaches in biology aim to understand short sequence windows of modified targets (motifs), there has not been a concerted effort to understand the long-range features that enable ISG15 selectivity. 

Deep learning enable representation learning over complex, multi-modal data common in protein biology. The purpose of this work is two-fold:

- Create a ISG15 modification prediction model.
- Parse learned representations for ISG15 target explainability insights.

---



## Input Data

This work leverages TMT-MS data produced in [Cellular targets and lysine selectivity of the HERC5 ISG15 ligase](https://www.cell.com/iscience/fulltext/S2589-0042(24)00041-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2589004224000415%3Fshowall%3Dtrue)

[EDA](data/eda/huibregtse_2024)

Briefly, experimental conditions that drive ISG15 activity were used on cell models with HERC5 knocked out, providing insights into what targets require HERC5 for modification. TMT-MS was used to specifically identified modified lysine residues with a left-over Glycine-Glycine structure (from ISG15) hanging off it. To ensure signal is not polluted by ubiquitin modification, the researchers used treatment with USP2 to abolish ubiquitylation.

---



## Pipeline

```
Raw MaxQuant files
        │
        ▼
   [1] prepare   — load, filter, merge, fetch FASTA, build lysine index
        │
        ▼
   [2] split     — homology-aware 70 / 15 / 15 train/val/test via graph-part (30 % threshold)
        │
        ▼
   [3] embed     — full-sequence ProtT5 (L × 1024) → compressed HDF5
        │
        ▼
   [4] train     — BucketSampler → BiLSTM → focal loss
        │
        ▼
   [5] evaluate  — per-site metrics on the held-out test split
```

> **Why split before embed?** ProtT5 is compute-intensive. graph-part runs on sequence identity alone and is cheap. Splitting first avoids embedding proteins that will be discarded.

---



## Model

`FullSequenceBiLSTMClassifier` in `src/models/bilstm.py`


| Hyperparameter       | Value                         |
| -------------------- | ----------------------------- |
| Input size           | 1024 (ProtT5 residue dim)     |
| Hidden size          | 256 (× 2 bidirectional → 512) |
| Layers               | 2                             |
| Dropout              | 0.5                           |
| Loss                 | Focal (α=1.0, γ=2.0)          |
| Optimiser            | AdamW lr=1e-4, wd=1e-3        |
| Effective batch size | 128                           |
| Epochs               | 50                            |


The forward pass packs the padded batch, runs the LSTM, then extracts the hidden state at `target_k_index` for each sample—the per-residue vector that represents that lysine in its full sequence context.

---



## Results (dev checkpoint)


| ACC   | BACC  | SN    | SP    | MCC   | Kappa | F1    | AUC   | AUPRC |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.923 | 0.582 | 0.185 | 0.978 | 0.231 | 0.215 | 0.250 | 0.763 | 0.233 |


The class imbalance is severe (most lysines are unmodified). AUC=0.76 and AUPRC=0.23 are the more informative headline numbers.

---



## Project layout

```
.
├── cli.py                  # Pipeline entry point (all stages)
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py       # Load raw MaxQuant files
│   │   ├── filter.py       # Remove low-quality / contaminant rows
│   │   ├── merge.py        # Build modified/unmodified corpus
│   │   ├── fasta.py        # Fetch sequences from UniProt
│   │   ├── lysine.py       # Lysine-index builder
│   │   ├── split.py        # Homology split @ 30% similarity
│   │   ├── embedder.py     # ProtT5 → HDF5
│   │   └── bucket_sampler.py  # Length-bucket DataLoader
│   ├── models/
│   │   └── bilstm.py       # FullSequenceBiLSTMClassifier
│   └── training/
│       ├── trainer.py      # Training loop + gradient accumulation
│       └── evaluator.py    # Test-set metrics
├── data/                   # Pipeline artefacts (gitignored)
├── checkpoints/            # Saved model weights (gitignored)
├── logs/                   # CSV training logs
└── results/
    └── metrics.json        # Latest evaluation output
```

---



## Quickstart

**Requirements:** Python 3.14, pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place raw MaxQuant output directories under `data/raw/`:

```
data/raw/
├── 8604_txt/
│   ├── GlyGly (K)Sites.txt
│   └── proteinGroups.txt
└── 8739_txt/
    ├── GlyGly (K)Sites.txt
    └── proteinGroups.txt
```

**Run the full pipeline:**

```bash
python cli.py run
```

**Run stages individually:**

```bash
python cli.py prepare
python cli.py split
python cli.py embed   --device cuda
python cli.py train   --device cuda --epochs 50
python cli.py evaluate
```

**Smoke-test on a small subset:**

```bash
python cli.py run --max-proteins 20
```

All stages are idempotent — rerunning `run` skips stages whose output files already exist. Use `--force` to override.

---



## CLI reference

```
python cli.py <command> --help
```


| Command    | Description                                                       |
| ---------- | ----------------------------------------------------------------- |
| `prepare`  | Load raw data → filter → merge → fetch FASTA → build lysine index |
| `split`    | Homology-aware split via graph-part                               |
| `embed`    | Embed proteins with ProtT5; write compressed HDF5                 |
| `train`    | Train BiLSTM; save best checkpoint by val AUPRC                   |
| `evaluate` | Score best checkpoint on the test split                           |
| `run`      | Full pipeline; respects existing artefacts unless `--force`       |

---

## Disclaimer

This project was built using AI. The architecture and requirements were pre-defined by a human, and all required work was detailed in `project_build`. Briefly, an agent was instructed to orchestrate a multi-wave implementation plan. The `project_build` folder can be used to rebuild this model.
