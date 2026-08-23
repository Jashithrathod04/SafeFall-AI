from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    CLASS_NAMES,
    BATCH_SIZE,
)

from src.model.activity_classifier import (
    ActivityClassifier
)


def main():

    data_dir = Path(
        PROCESSED_DATA_DIR
    )

    X = np.load(
        data_dir /
        "X_test.npy"
    )

    y = np.load(
        data_dir /
        "y_test.npy"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ActivityClassifier(
        num_classes=len(
            CLASS_NAMES
        )
    )

    checkpoint = torch.load(
        MODEL_DIR /
        "best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.long
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)

            outputs = model(
                inputs
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            y_pred.extend(
                predictions
            )

            y_true.extend(
                labels.numpy()
            )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(
                len(CLASS_NAMES)
            ),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )


if __name__ == "__main__":
    main()
