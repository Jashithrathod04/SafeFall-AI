from pathlib import Path

import numpy as np

from src.config import (
    KEYPOINTS_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
)


class SequenceGenerator:

    def __init__(self):

        self.keypoints_dir = Path(
            KEYPOINTS_DIR
        )

        self.output_dir = Path(
            PROCESSED_DATA_DIR
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_video_keypoints(self, video_dir):

        files = sorted(
            video_dir.glob("*.npy")
        )

        if not files:
            return np.empty(
                (0, 51),
                dtype=np.float32
            )

        data = []

        for file in files:

            values = np.load(file)

            data.append(values)

        return np.asarray(
            data,
            dtype=np.float32
        )

    def generate_sequences(
        self,
        data,
        label
    ):

        X = []
        y = []

        if len(data) < SEQUENCE_LENGTH:
            return X, y

        for start in range(
            0,
            len(data) - SEQUENCE_LENGTH + 1,
            SEQUENCE_STRIDE
        ):

            sequence = data[
                start:
                start + SEQUENCE_LENGTH
            ]

            X.append(sequence)
            y.append(label)

        return X, y

    def create_sequences(
        self,
        labelled_video_dirs
    ):

        X = []
        y = []

        for video_dir, label in labelled_video_dirs:

            data = self.load_video_keypoints(
                video_dir
            )

            sequences, labels = (
                self.generate_sequences(
                    data,
                    label
                )
            )

            X.extend(sequences)
            y.extend(labels)

        if not X:

            raise RuntimeError(
                "No sequences were generated."
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.int64
        )

        np.save(
            self.output_dir /
            "X_sequences.npy",
            X
        )

        np.save(
            self.output_dir /
            "y_labels.npy",
            y
        )

        print(
            "X shape:",
            X.shape
        )

        print(
            "y shape:",
            y.shape
        )

        return X, y
