import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import (
    CLASS_NAMES,
    SEQUENCE_LENGTH,
)

from src.pose.pose_detector import (
    PoseDetector
)

from src.model.predictor import (
    Predictor
)

from src.detection.temporal_validator import (
    TemporalValidator
)

from src.alerts.alert_manager import (
    AlertManager
)


st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title(
    "🛡️ SafeFall AI"
)

st.subheader(
    "AI-Powered Elderly Fall Detection System"
)

st.write(
    "Computer vision + YOLOv8 Pose + "
    "BiLSTM temporal activity recognition."
)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_system():

    pose_detector = PoseDetector()

    predictor = Predictor()

    validator = TemporalValidator()

    alert_manager = AlertManager()

    return (
        pose_detector,
        predictor,
        validator,
        alert_manager,
    )


try:

    (
        pose_detector,
        predictor,
        validator,
        alert_manager,
    ) = load_system()

except Exception as error:

    st.error(
        "The AI model could not be loaded."
    )

    st.exception(error)

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header(
    "Monitoring Settings"
)

confidence_threshold = st.sidebar.slider(
    "Fall confidence threshold",
    0.50,
    0.99,
    0.70,
    0.01
)

st.sidebar.info(
    "The system uses temporal confirmation "
    "to reduce false alarms."
)


# ---------------------------------------------------------
# VIDEO INPUT
# ---------------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload a surveillance video",
    type=["avi", "mp4", "mov"]
)


if uploaded_video is not None:

    st.video(
        uploaded_video
    )

    if st.button(
        "▶ Analyze Video"
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temporary:

            temporary.write(
                uploaded_video.read()
            )

            video_path = (
                temporary.name
            )

        capture = cv2.VideoCapture(
            video_path
        )

        sequence = []

        prediction_placeholder = (
            st.empty()
        )

        progress = st.progress(0)

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            frame_count = 1

        frame_index = 0

        detected_events = []

        while True:

            success, frame = (
                capture.read()
            )

            if not success:
                break

            keypoints = (
                pose_detector
                .extract_keypoints(
                    frame
                )
            )

            sequence.append(
                keypoints
            )

            if len(sequence) >= SEQUENCE_LENGTH:

                window = np.asarray(
                    sequence[
                        -SEQUENCE_LENGTH:
                    ],
                    dtype=np.float32
                )

                result = (
                    predictor.predict(
                        window
                    )
                )

                label = result[
                    "label"
                ]

                confidence = result[
                    "confidence"
                ]

                prediction_placeholder.metric(
                    "Current Activity",
                    label,
                    f"{confidence * 100:.1f}% confidence"
                )

                confirmed = (
                    validator.update(
                        label,
                        confidence
                    )
                )

                if confirmed:

                    alert = (
                        alert_manager
                        .trigger()
                    )

                    detected_events.append(
                        alert
                    )

            frame_index += 1

            progress.progress(
                min(
                    frame_index /
                    frame_count,
                    1.0
                )
            )

        capture.release()

        st.divider()

        if detected_events:

            st.error(
                "🚨 FALL DETECTED"
            )

            st.write(
                detected_events[-1]
            )

        else:

            st.success(
                "No confirmed fall detected."
            )
