from pathlib import Path


# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
ASSET_DIR = ROOT_DIR / "assets"


# Activity classes required by the assignment
CLASSES = [
    "Fall Detected",
    "Walking",
    "Sitting",
    "Standing",
    "Normal Activity"
]


# Pose configuration
NUM_KEYPOINTS = 17
FEATURES_PER_KEYPOINT = 3

# x + y + confidence
NUM_FEATURES = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT


# Temporal sequence
SEQUENCE_LENGTH = 30


# Image preprocessing
IMAGE_SIZE = (224, 224)
