from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    POSE_MODEL,
    NUM_FEATURES,
)


class PoseDetector:

    def __init__(self, model_path=POSE_MODEL):

        self.model = YOLO(model_path)

    def extract_keypoints(self, image):

        if isinstance(image, (str, Path)):

            frame = cv2.imread(str(image))

        else:

            frame = image

        if frame is None:

            return np.zeros(
                NUM_FEATURES,
                dtype=np.float32
            )

        results = self.model(
            frame,
            verbose=False
        )

        if not results:
            return np.zeros(
                NUM_FEATURES,
                dtype=np.float32
            )

        result = results[0]

        if result.keypoints is None:
            return np.zeros(
                NUM_FEATURES,
                dtype=np.float32
            )

        if result.keypoints.data is None:
            return np.zeros(
                NUM_FEATURES,
                dtype=np.float32
            )

        keypoints = (
            result.keypoints.data[0]
            .cpu()
            .numpy()
        )

        keypoints = keypoints.astype(
            np.float32
        )

        # Normalize x and y coordinates
        height, width = frame.shape[:2]

        keypoints[:, 0] /= width
        keypoints[:, 1] /= height

        return keypoints.flatten()
