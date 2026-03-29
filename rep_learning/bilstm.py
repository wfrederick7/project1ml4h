import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=128, num_layers=2, proj_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # hidden_dim * 2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # projection head (used only during pretraining)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, proj_dim),
        )

    def encode(self, x):
        """
        x: (batch, seq_len, 41)
        Returns: (batch, 256) embedding
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # Last layer forward: h_n[-2], last layer backward: h_n[-1]
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)
        return self.fc(h_cat)

    def project(self, x):
        """
        Encode then project through the projection head.
        Returns L2-normalised (batch, 64) vectors.
        """
        emb = self.encode(x)
        z = self.proj_head(emb)
        return F.normalize(z, dim=1)

    def forward(self, x):
        return self.encode(x)
