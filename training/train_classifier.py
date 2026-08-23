from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    NUM_FEATURES,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)

from src.model.activity_classifier import (
    ActivityClassifier
)


def main():

    torch.manual_seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    data_dir = Path(
        PROCESSED_DATA_DIR
    )

    X_path = (
        data_dir /
        "X_sequences.npy"
    )

    y_path = (
        data_dir /
        "y_labels.npy"
    )

    if not X_path.exists():
        raise FileNotFoundError(
            f"Missing {X_path}"
        )

    X = np.load(X_path)
    y = np.load(y_path)

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset) -
        train_size
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [train_size, val_size]
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    model = ActivityClassifier(
        input_size=NUM_FEATURES,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += (
                labels.size(0)
            )

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                predictions = (
                    outputs.argmax(dim=1)
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += (
                    labels.size(0)
                )

        train_loss /= train_total
        val_loss /= val_total

        train_accuracy = (
            train_correct /
            train_total
        )

        val_accuracy = (
            val_correct /
            val_total
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),
                "val_loss":
                    val_loss,
                "val_accuracy":
                    val_accuracy,
            }

            torch.save(
                checkpoint,
                MODEL_DIR /
                "best_model.pt"
            )

            print(
                "  ✓ Best model saved"
            )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    main()
