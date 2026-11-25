"""Torch-only network used by B4Detector (kept separate for optional dependencies)."""

import torch
from torch import nn


class B4Network(nn.Module):
    def __init__(self, encoder, n_features):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(encoder.config.hidden_size + n_features, 1)

    def forward(self, input_ids, attention_mask, features):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        combined = torch.cat([self.dropout(hidden), features.to(hidden.dtype)], dim=1)
        return self.classifier(combined).squeeze(-1).float()
