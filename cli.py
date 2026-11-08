#!/usr/bin/env python3
"""
ISG15 target-prediction pipeline CLI.

Usage:
    python cli.py prepare   [options]
    python cli.py split     [options]
    python cli.py embed     [options]
    python cli.py train     [options]
    python cli.py evaluate  [options]
    python cli.py run       [options]   # full pipeline, stages skipped if outputs exist
"""

import argparse
import pathlib
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RAW_DATASETS = ["8604_txt", "8739_txt"]  # dirs that have GlyGly(K)Sites.txt


def _resolve(path: str) -> pathlib.Path:
    return pathlib.Path(path).resolve()

def _device(choice: str):
    import torch
    try:
        torch.device(choice)
    except (RuntimeError, ValueError):
        sys.exit(f"Error: --device {choice} requested but is not available.")
    return torch.device(choice)


def _make_run_dir(model_name: str) -> pathlib.Path:
    """Create and return a unique timestamped run directory under runs/."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("runs") / f"{stamp}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _require(path: pathlib.Path, produced_by: str) -> None:
    """Abort with a clear message if a required input file is missing."""
    if not path.exists():
        sys.exit(f"Error: {path} not found. Run `python cli.py {produced_by}` first.")


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    import pickle
    from src.data.loader import load_raw_data
    from src.data.filter import filter_evidence, filter_ksites
    from src.data.merge import build_corpus
    from src.data.fasta import retrieve_fasta
    from src.data.lysine import build_lysine_index, parse_fasta

    raw_dirs = [_resolve(args.raw_dir) / d for d in RAW_DATASETS]
    for d in raw_dirs:
        if not d.exists():
            sys.exit(f"Error: raw dataset directory not found: {d}")

    fasta_out   = _resolve(args.fasta_out)
    corpus_out  = _resolve(args.corpus_out)
    lysine_out  = _resolve(args.lysine_out)

    print("── prepare ──────────────────────────────────────")
    df_ev, df_ks = load_raw_data(raw_dirs)
    print(f"Loaded {len(df_ev):,} evidence rows, {len(df_ks):,} k-site rows")

    df_ev_clean = filter_evidence(df_ev)
    df_ks_clean = filter_ksites(df_ks)
    print(f"After filtering: {len(df_ev_clean):,} evidence, {len(df_ks_clean):,} k-sites")

    df_modified, df_unmodified = build_corpus(df_ev_clean, df_ks_clean)
    print(f"Corpus: {df_modified['Protein'].nunique():,} modified proteins, "
          f"{len(df_unmodified):,} unmodified rows")

    protein_ids = df_ks_clean["Protein"].unique().tolist()

    if args.max_proteins is not None:
        protein_ids = protein_ids[: args.max_proteins]
        df_modified  = df_modified[df_modified["Protein"].isin(protein_ids)].reset_index(drop=True)
        df_unmodified = df_unmodified[
            df_unmodified["Proteins"].apply(lambda ps: any(p in protein_ids for p in ps))
        ].reset_index(drop=True)
        print(f"Subset: capped at {args.max_proteins} proteins "
              f"({df_modified['Protein'].nunique():,} modified, "
              f"{len(df_unmodified):,} unmodified rows retained)")

    retrieve_fasta(protein_ids, fasta_out)
    print(f"FASTA → {fasta_out}  ({len(protein_ids):,} sequences)")

    sequences = parse_fasta(fasta_out)
    df_corpus = build_lysine_index(df_modified, df_unmodified, sequences)
    n_pos = (df_corpus["label"] == 1).sum()
    n_neg = (df_corpus["label"] == 0).sum()
    print(f"Lysine corpus: {len(df_corpus):,} sites  ({n_pos:,} positive, {n_neg:,} negative)")

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    lysine_out.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_out, "wb") as f:
        pickle.dump((df_modified, df_unmodified), f)
    with open(lysine_out, "wb") as f:
        pickle.dump(df_corpus, f)
    print(f"Corpus  → {corpus_out}")
    print(f"Lysine  → {lysine_out}")


def cmd_split(args):
    from src.data.split import split_by_homology

    fasta_in  = _resolve(args.fasta_in)
    split_out = _resolve(args.split_out)
    _require(fasta_in, "prepare")

    print("── split ────────────────────────────────────────")
    split_assignments = split_by_homology(fasta_in, split_out)
    counts = split_assignments["split"].value_counts()
    print(f"Split → {split_out}")
    print(f"  train {counts.get('train', 0)}  |  val {counts.get('val', 0)}  |  test {counts.get('test', 0)}")


def cmd_embed(args):
    import pickle
    from src.data.embedder import embed_sequences

    fasta_in       = _resolve(args.fasta_in)
    split_in       = _resolve(args.split_in)
    lysine_in      = _resolve(args.lysine_in)
    embeddings_out = _resolve(args.embeddings_out)

    _require(fasta_in,  "prepare")
    _require(split_in,  "split")
    _require(lysine_in, "prepare")

    import pandas as pd
    split_assignments = pd.read_csv(split_in)
    with open(lysine_in, "rb") as f:
        df_corpus = pickle.load(f)

    device = _device(args.device)
    print(f"── embed  (device: {device}) ─────────────────────")
    embed_sequences(fasta_in, split_assignments, df_corpus, embeddings_out)
    print(f"Embeddings → {embeddings_out}")


def _build_model(model_name: str):
    """Instantiate a model by name. Raises on unknown names (fail fast)."""
    from src.models.bilstm import LysineEmbeddingClassifier, FullSequenceBiLSTMClassifier
    models = {
        "linear": LysineEmbeddingClassifier,
        "bilstm": FullSequenceBiLSTMClassifier,
    }
    if model_name not in models:
        sys.exit(f"Error: unknown --model '{model_name}'. Choose from: {list(models)}")
    return models[model_name]()


def cmd_train(args):
    from src.data.bucket_sampler import create_dataloader
    from src.training.trainer import train, TrainConfig

    embeddings_in = _resolve(args.embeddings_in)
    _require(embeddings_in, "embed")

    # Auto-generate a timestamped run directory unless the caller supplied
    # explicit paths (i.e. they are still None / the sentinel default).
    if args.checkpoint_dir is None and args.log_dir is None:
        run_dir = _make_run_dir(args.model)
        checkpoint_dir = run_dir
        log_dir        = run_dir
        # Stash so cmd_run can pass it straight to cmd_evaluate.
        args._run_dir = run_dir
        print(f"Run directory: {run_dir}")
    else:
        checkpoint_dir = _resolve(args.checkpoint_dir or "checkpoints")
        log_dir        = _resolve(args.log_dir        or "logs")
        args._run_dir  = None

    device = _device(args.device)
    print(f"── train  (device: {device}, model: {args.model}) ──────────")

    train_loader = create_dataloader(embeddings_in, "train", args.batch_size, args.bucket_width)
    val_loader   = create_dataloader(embeddings_in, "val",   args.batch_size, args.bucket_width)

    model = _build_model(args.model).to(device)
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        effective_batch_size=args.effective_batch_size,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    train(model, train_loader, val_loader, cfg)


def cmd_evaluate(args):
    from src.data.bucket_sampler import create_dataloader
    from src.training.evaluator import evaluate

    embeddings_in   = _resolve(args.embeddings_in)
    checkpoint_path = _resolve(args.checkpoint)
    _require(embeddings_in,   "embed")
    _require(checkpoint_path, "train")

    device = _device(args.device)
    print(f"── evaluate  (device: {device}, model: {args.model}) ────────")

    # Write metrics.json into the same directory as the checkpoint so every
    # run is self-contained. Fall back to results/metrics.json only when the
    # checkpoint lives in the legacy default location.
    results_path = getattr(args, "_results_path", None)
    if results_path is None:
        results_path = checkpoint_path.parent / "metrics.json"

    test_loader = create_dataloader(embeddings_in, "test", args.batch_size, args.bucket_width)
    model = _build_model(args.model).to(device)
    metrics = evaluate(model, test_loader, checkpoint_path, results_path=results_path)

    print(f"\nTest-set metrics (written to {results_path}):")
    print("  " + "  |  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def cmd_run(args):
    """Full pipeline. Each stage is skipped if its output already exists."""

    # Wire stage outputs → next stage's inputs so a single custom path
    # propagates through the whole pipeline (DRY: out IS in for run).
    args.fasta_in      = args.fasta_out
    args.split_in      = args.split_out
    args.lysine_in     = args.lysine_out
    args.embeddings_in = args.embeddings_out

    def _exists(path: str) -> bool:
        return _resolve(path).exists()

    if args.force or not _exists(args.fasta_out):
        # prepare doesn't use a run dir; only train/evaluate do
        cmd_prepare(args)

    else:
        print(f"prepare: skipped  ({args.fasta_out} exists)")

    if args.force or not _exists(args.split_out):
        cmd_split(args)
    else:
        print(f"split:   skipped  ({args.split_out} exists)")

    if args.force or not _exists(args.embeddings_out):
        cmd_embed(args)
    else:
        print(f"embed:   skipped  ({args.embeddings_out} exists)")

    if args.force or not _exists(args.checkpoint):
        cmd_train(args)
        # cmd_train sets args._run_dir; derive checkpoint path from it.
        if args._run_dir is not None:
            args.checkpoint    = str(args._run_dir / "best_model.pt")
            args._results_path = args._run_dir / "metrics.json"
    else:
        print(f"train:   skipped  ({args.checkpoint} exists)")

    cmd_evaluate(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_data_dir(p, default="data"):
    p.add_argument("--data-dir", default=default, metavar="DIR",
                   help="Root directory for all pipeline artefacts (default: %(default)s)")

def _add_raw_dir(p):
    p.add_argument("--raw-dir", default="data/raw", metavar="DIR",
                   help="Directory containing MaxQuant dataset folders (default: %(default)s)")

def _add_device(p):
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device for model inference/training (default: %(default)s)")

def _add_model(p):
    p.add_argument("--model", default="linear", choices=["linear", "bilstm"],
                   help="Model architecture: 'linear' = ProtT5 embedding → FC head (baseline); "
                        "'bilstm' = BiLSTM over full sequence → FC head (default: %(default)s)")

def _add_loader_opts(p):
    p.add_argument("--batch-size", type=int, default=32, metavar="N",
                   help="Physical batch size per step (default: %(default)s)")
    p.add_argument("--bucket-width", type=int, default=50, metavar="N",
                   help="Sequence-length bucket width for BucketSampler (default: %(default)s)")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python cli.py",
        description="ISG15 target-prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── prepare ──────────────────────────────────────────────────────
    p = sub.add_parser("prepare", help="Load, filter, merge, fetch FASTA, build lysine index",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_raw_dir(p)
    p.add_argument("--fasta-out",  default="data/proteins.fasta",    metavar="FILE")
    p.add_argument("--corpus-out", default="data/corpus.pkl",        metavar="FILE")
    p.add_argument("--lysine-out", default="data/lysine_corpus.pkl", metavar="FILE")
    p.add_argument("--max-proteins", type=int, default=None, metavar="N",
                   help="Cap the protein set to N proteins after filtering. "
                        "Useful for smoke-testing the full pipeline quickly.")
    p.set_defaults(func=cmd_prepare)

    # ── split ─────────────────────────────────────────────────────────
    p = sub.add_parser("split", help="Homology-aware train/val/test split via graph-part",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fasta-in",  default="data/proteins.fasta",       metavar="FILE")
    p.add_argument("--split-out", default="data/split_assignments.csv", metavar="FILE")
    p.set_defaults(func=cmd_split)

    # ── embed ─────────────────────────────────────────────────────────
    p = sub.add_parser("embed", help="Embed proteins with ProtT5 → embeddings.h5",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fasta-in",       default="data/proteins.fasta",        metavar="FILE")
    p.add_argument("--split-in",       default="data/split_assignments.csv", metavar="FILE")
    p.add_argument("--lysine-in",      default="data/lysine_corpus.pkl",     metavar="FILE")
    p.add_argument("--embeddings-out", default="data/embeddings.h5",         metavar="FILE")
    _add_device(p)
    p.set_defaults(func=cmd_embed)

    # ── train ─────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Train the BiLSTM classifier",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--embeddings-in",      default="data/embeddings.h5",  metavar="FILE")
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--effective-batch-size", type=int, default=128,
                   help="Target batch size for gradient accumulation")
    p.add_argument("--checkpoint-dir", default=None, metavar="DIR",
                   help="Directory for best_model.pt. Defaults to a new runs/<timestamp>_<model>/ directory.")
    p.add_argument("--log-dir",        default=None, metavar="DIR",
                   help="Directory for train_log.csv. Defaults to the same runs/ directory as --checkpoint-dir.")
    _add_device(p)
    _add_model(p)
    _add_loader_opts(p)
    p.set_defaults(func=cmd_train)

    # ── evaluate ──────────────────────────────────────────────────────
    p = sub.add_parser("evaluate", help="Evaluate best checkpoint on the test split",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--embeddings-in", default="data/embeddings.h5",         metavar="FILE")
    p.add_argument("--checkpoint",    default="checkpoints/best_model.pt",  metavar="FILE")
    _add_device(p)
    _add_model(p)
    _add_loader_opts(p)
    p.set_defaults(func=cmd_evaluate)

    # ── run ───────────────────────────────────────────────────────────
    p = sub.add_parser("run", help="Full pipeline (stages skipped if outputs exist)",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_raw_dir(p)
    p.add_argument("--fasta-out",          default="data/proteins.fasta",        metavar="FILE")
    p.add_argument("--corpus-out",         default="data/corpus.pkl",            metavar="FILE")
    p.add_argument("--lysine-out",         default="data/lysine_corpus.pkl",     metavar="FILE")
    p.add_argument("--max-proteins", type=int, default=None, metavar="N",
                   help="Cap the protein set to N proteins (passed through to prepare)")
    p.add_argument("--split-out",          default="data/split_assignments.csv", metavar="FILE")
    p.add_argument("--embeddings-out",     default="data/embeddings.h5",         metavar="FILE")
    p.add_argument("--fasta-in",           default="data/proteins.fasta",        metavar="FILE")
    p.add_argument("--split-in",           default="data/split_assignments.csv", metavar="FILE")
    p.add_argument("--lysine-in",          default="data/lysine_corpus.pkl",     metavar="FILE")
    p.add_argument("--embeddings-in",      default="data/embeddings.h5",         metavar="FILE")
    p.add_argument("--checkpoint",     default="checkpoints/best_model.pt", metavar="FILE")
    p.add_argument("--checkpoint-dir", default=None, metavar="DIR",
                   help="Override run directory for checkpoint. Defaults to runs/<timestamp>_<model>/.")
    p.add_argument("--log-dir",        default=None, metavar="DIR",
                   help="Override run directory for logs. Defaults to same as --checkpoint-dir.")
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--effective-batch-size", type=int, default=128)
    _add_device(p)
    _add_model(p)
    _add_loader_opts(p)
    p.add_argument("--force", action="store_true",
                   help="Re-run all stages even if output artefacts exist")
    p.set_defaults(func=cmd_run)

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
