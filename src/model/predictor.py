from pathlib import Path

import numpy as np
import torch

from src.config import (
    CLASS_NAMES,
    MODEL_DIR,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)

from src.model.activity_classifier import (
    ActivityClassifier
)


class Predictor:

    def __init__(self, model_path=None):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = ActivityClassifier()

        if model_path is None:
            model_path = (
                Path(MODEL_DIR) /
                "best_model.pt"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        if isinstance(
            checkpoint,
            dict
        ) and "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, sequence):

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            NUM_FEATURES
        ):

            raise ValueError(
                f"Expected shape "
                f"({SEQUENCE_LENGTH}, "
                f"{NUM_FEATURES}), "
                f"got {sequence.shape}"
            )

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

        label = CLASS_NAMES[
            prediction.item()
        ]

        return {
            "label": label,
            "class_id": prediction.item(),
            "confidence": confidence.item(),
            "probabilities":
                probabilities[0]
                .cpu()
                .numpy(),
        }
