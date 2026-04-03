import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder matching the Q2.2 architecture.

    Encoder: BiLSTM -> dropout -> recency-weighted pooling -> embedding
    Projection head (contrastive pretraining only): embedding -> MLP -> L2-norm
    """

    def __init__(self, input_dim=41, hidden_dim=64, num_layers=1,
                 dropout=0.2, recency_strength=2.0, proj_dim=64):
        super().__init__()
        self.recency_strength = recency_strength

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        out_dim = hidden_dim * 2  # bidirectional
        self.dropout = nn.Dropout(dropout)

        # Projection head (used only during contrastive pretraining)
        self.proj_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, proj_dim),
        )

    def _recency_weights(self, T, device):
        """Exponential recency weights matching Q2.2."""
        t = torch.linspace(0, 1, T, device=device)
        w = torch.exp(self.recency_strength * t)
        return w / w.sum()

    def encode(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, hidden_dim*2) embedding via recency-weighted pooling.
        """
        h, _ = self.lstm(x)                        # (B, T, hidden_dim*2)
        h = self.dropout(h)                         # (B, T, hidden_dim*2)
        w = self._recency_weights(h.size(1), h.device)
        emb = (h * w.view(1, -1, 1)).sum(dim=1)    # (B, hidden_dim*2)
        return emb

    def project(self, x):
        """Encode then project. Returns L2-normalised (batch, proj_dim) vectors."""
        emb = self.encode(x)
        z = self.proj_head(emb)
        return F.normalize(z, dim=1)

    def forward(self, x):
        return self.encode(x)
