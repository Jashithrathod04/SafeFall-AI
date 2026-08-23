import torch
import torch.nn as nn

from src.config import (
    NUM_FEATURES,
    NUM_CLASSES,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
)


class ActivityClassifier(nn.Module):

    def __init__(
        self,
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0
            ),
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                num_classes
            )
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use final temporal representation
        features = output[:, -1, :]

        return self.classifier(features)
