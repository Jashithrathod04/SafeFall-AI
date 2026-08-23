from pathlib import Path

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models" / "activity_model"
OUTPUT_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

CLASS_NAMES = [
    "Fall Detected",
    "Walking",
    "Sitting",
    "Standing",
    "Normal Activity",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------
# POSE
# ---------------------------------------------------------

POSE_MODEL = "yolov8n-pose.pt"

NUM_KEYPOINTS = 17
FEATURES_PER_KEYPOINT = 3

# x, y, confidence
NUM_FEATURES = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT

# ---------------------------------------------------------
# VIDEO / TEMPORAL SETTINGS
# ---------------------------------------------------------

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

SEQUENCE_LENGTH = 30
SEQUENCE_STRIDE = 10

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 1e-3

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# ---------------------------------------------------------
# FALL VALIDATION
# ---------------------------------------------------------

FALL_CONFIDENCE_THRESHOLD = 0.70
FALL_CONFIRMATION_WINDOWS = 2

# ---------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------

RANDOM_SEED = 42
