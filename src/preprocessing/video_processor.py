
from pathlib import Path

import cv2


class VideoProcessor:

    def __init__(self, output_size=(224, 224), sample_every=1):
        self.output_size = output_size
        self.sample_every = sample_every

    def extract_frames(self, video_path, output_dir):

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_index = 0
        saved_count = 0

        while True:

            success, frame = capture.read()

            if not success:
                break

            if frame_index % self.sample_every == 0:

                frame = cv2.resize(
                    frame,
                    self.output_size
                )

                output_file = (
                    output_dir /
                    f"frame_{frame_index:06d}.jpg"
                )

                cv2.imwrite(
                    str(output_file),
                    frame
                )

                saved_count += 1

            frame_index += 1

        capture.release()

        return saved_count
