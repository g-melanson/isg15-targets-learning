"""Training loop for FullSequenceBiLSTMClassifier.

Implements:
- FocalLoss: binary focal loss FL(p) = -alpha * (1-p)^gamma * log(p)
- TrainConfig: dataclass holding all hyperparameters
- train(): full training loop with gradient accumulation, AUPRC checkpointing,
  and per-epoch CSV logging
"""

import csv
import math
import pathlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss applied to raw logits.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    where p = sigmoid(logit).  Downweights easy negatives so training focuses
    on hard examples.

    Args:
        alpha: Scaling factor for the focal term (default 1.0).
        gamma: Focusing exponent; higher → more focus on hard examples (default 2.0).
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss.

        Args:
            logits: Raw model outputs, shape (B,).
            targets: Binary labels as float32, shape (B,).

        Returns:
            Scalar mean loss.
        """
        probs = torch.sigmoid(logits)
        # Clamp to avoid log(0) → -inf
        probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)

        # For positive targets: -alpha * (1-p)^gamma * log(p)
        # For negative targets: -alpha * p^gamma * log(1-p)
        loss_pos = -self.alpha * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        loss_neg = -self.alpha * (probs ** self.gamma) * torch.log(1.0 - probs)

        loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return loss.mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyperparameters for the training loop.

    All fields have sensible defaults so callers only need to override what they
    actually want to change (YAGNI — no unused knobs).
    """

    loss: str = "FocalLoss"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    epochs: int = 50
    effective_batch_size: int = 128
    checkpoint_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("checkpoints"))
    log_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("logs"))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_CSV_COLUMNS = ["epoch", "train_loss", "val_loss", "train_auprc", "val_auprc"]


def train(model, train_loader, val_loader, cfg: TrainConfig) -> None:
    """Run the full training loop.

    Gradient accumulation: accumulates gradients for
    ``ceil(cfg.effective_batch_size / physical_batch_size)`` micro-steps before
    calling ``optimizer.step()``.  This keeps the effective batch size constant
    regardless of how many samples actually fit in memory.

    Checkpointing: saves ``cfg.checkpoint_dir/best_model.pt`` whenever val AUPRC
    exceeds the best seen so far (overwrites in place).

    Logging: appends one CSV row per epoch to ``cfg.log_dir/train_log.csv``.

    Args:
        model: FullSequenceBiLSTMClassifier (or any module with compatible forward).
        train_loader: DataLoader yielding (x, seq_lengths, target_k_indices, labels).
        val_loader:   DataLoader yielding (x, seq_lengths, target_k_indices, labels).
        cfg:          TrainConfig with all hyperparameters.
    """
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    log_path = cfg.log_dir / "train_log.csv"
    # Write header only if file doesn't exist yet (so resuming doesn't duplicate it)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_COLUMNS).writeheader()

    best_val_auprc = -1.0

    # Gradient accumulation steps is fixed for the whole run: based on a
    # full-sized batch so the last (potentially smaller) batch doesn't
    # inflate accum_steps to a nonsensical value (YAGNI / no magic numbers).
    accum_steps = max(1, math.ceil(cfg.effective_batch_size / train_loader.batch_sampler.batch_size))

    for epoch in range(1, cfg.epochs + 1):
        # ------------------------------------------------------------------ #
        # Training phase
        # ------------------------------------------------------------------ #
        model.train()
        train_loss_sum = 0.0
        train_all_logits: list[torch.Tensor] = []
        train_all_labels: list[torch.Tensor] = []

        optimizer.zero_grad()
        step_in_accum = 0  # counts micro-steps within the current accumulation window

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{cfg.epochs:03d} [train]",
            leave=False,
        )
        for x, seq_lengths, target_k_indices, labels in train_bar:
            x = x.to(device)
            seq_lengths = seq_lengths.to(device)
            target_k_indices = target_k_indices.to(device)
            labels = labels.to(device)

            logits = model(x, seq_lengths, target_k_indices)  # (B, 1)
            logits = logits.squeeze(1)                         # (B,)

            # Scale loss so that summed gradients equal the mean over the
            # effective batch (gradient accumulation standard practice).
            loss = criterion(logits, labels) / accum_steps
            loss.backward()

            train_loss_sum += loss.item() * accum_steps  # unscale for logging
            train_all_logits.append(logits.detach().cpu())
            train_all_labels.append(labels.detach().cpu())

            step_in_accum += 1
            if step_in_accum >= accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                step_in_accum = 0

        # Flush any remaining accumulated gradients at end of epoch
        if step_in_accum > 0:
            optimizer.step()
            optimizer.zero_grad()

        num_train_batches = len(train_loader)
        train_loss = train_loss_sum / num_train_batches if num_train_batches > 0 else 0.0

        all_train_logits = torch.cat(train_all_logits).numpy()
        all_train_labels = torch.cat(train_all_labels).numpy()
        train_probs = torch.sigmoid(torch.from_numpy(all_train_logits)).numpy()
        train_auprc = float(average_precision_score(all_train_labels, train_probs))

        # ------------------------------------------------------------------ #
        # Validation phase
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss_sum = 0.0
        val_all_logits: list[torch.Tensor] = []
        val_all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x, seq_lengths, target_k_indices, labels in val_loader:
                x = x.to(device)
                seq_lengths = seq_lengths.to(device)
                target_k_indices = target_k_indices.to(device)
                labels = labels.to(device)

                logits = model(x, seq_lengths, target_k_indices).squeeze(1)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                val_all_logits.append(logits.cpu())
                val_all_labels.append(labels.cpu())

        num_val_batches = len(val_loader)
        val_loss = val_loss_sum / num_val_batches if num_val_batches > 0 else 0.0

        all_val_logits = torch.cat(val_all_logits).numpy()
        all_val_labels = torch.cat(val_all_labels).numpy()
        val_probs = torch.sigmoid(torch.from_numpy(all_val_logits)).numpy()
        val_auprc = float(average_precision_score(all_val_labels, val_probs))

        # ------------------------------------------------------------------ #
        # Checkpointing
        # ------------------------------------------------------------------ #
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            checkpoint_path = cfg.checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)

        # ------------------------------------------------------------------ #
        # Logging
        # ------------------------------------------------------------------ #
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "train_auprc": f"{train_auprc:.6f}",
            "val_auprc": f"{val_auprc:.6f}",
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_COLUMNS).writerow(row)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"train_auprc={train_auprc:.4f}  val_auprc={val_auprc:.4f}"
            + (" ✓ saved" if val_auprc == best_val_auprc else "")
        )
