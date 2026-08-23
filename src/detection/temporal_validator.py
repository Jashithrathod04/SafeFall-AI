from collections import deque

from src.config import (
    FALL_CONFIDENCE_THRESHOLD,
    FALL_CONFIRMATION_WINDOWS,
)


class TemporalValidator:

    def __init__(self):

        self.history = deque(
            maxlen=FALL_CONFIRMATION_WINDOWS
        )

    def update(
        self,
        label,
        confidence
    ):

        is_fall = (
            label == "Fall Detected"
            and confidence >=
            FALL_CONFIDENCE_THRESHOLD
        )

        self.history.append(
            is_fall
        )

        confirmed = (
            len(self.history) ==
            FALL_CONFIRMATION_WINDOWS
            and all(self.history)
        )

        return confirmed

    def reset(self):

        self.history.clear()
