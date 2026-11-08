"""Classifiers for ISGylation site prediction.

Two models with the same forward signature (x, seq_lengths, target_k_indices):

- LysineEmbeddingClassifier: linear probe directly on the ProtT5 embedding at
  the target lysine. No sequential context. Use as the baseline.

- FullSequenceBiLSTMClassifier: runs the full sequence through a BiLSTM then
  extracts the hidden state at the target lysine. Adds learned sequential
  context on top of the ProtT5 embedding.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LysineEmbeddingClassifier(nn.Module):
    """Linear probe on the ProtT5 embedding at the target lysine position.

    No sequential context is added — the model classifies each lysine
    solely from the 1024-dim ProtT5 vector at that position.  Use this as
    the baseline before introducing recurrent context.

    Accepts the same forward signature as FullSequenceBiLSTMClassifier so
    both models are drop-in replacements for each other.
    """

    def __init__(self, input_size: int = 1024) -> None:
        super().__init__()
        self.fc_head = nn.Linear(input_size, 1)

    def forward(
        self,
        x: torch.Tensor,                # (B, L_max, 1024) padded ProtT5 embeddings
        seq_lengths: torch.Tensor,       # (B,) — accepted but unused
        target_k_indices: torch.Tensor,  # (B,) 0-indexed positions of target lysine
    ) -> torch.Tensor:                   # (B, 1) raw logits
        B = x.size(0)
        lysine_emb = x[torch.arange(B), target_k_indices]  # (B, 1024)
        return self.fc_head(lysine_emb)                     # (B, 1)


class FullSequenceBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc_head = nn.Linear(lstm_output_size, 1)

    def forward(
        self,
        x: torch.Tensor,               # (B, L_max, 1024) padded ProtT5 embeddings
        seq_lengths: torch.Tensor,      # (B,) actual unpadded lengths
        target_k_indices: torch.Tensor, # (B,) 0-indexed positions of target lysine
    ) -> torch.Tensor:                  # (B, 1) raw logits (no sigmoid)
        B = x.size(0)

        packed = pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Extract hidden state at each sample's target lysine position
        lysine_hidden = lstm_out[torch.arange(B), target_k_indices]  # (B, 512)

        logits = self.fc_head(lysine_hidden)  # (B, 1)
        return logits
