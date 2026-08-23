from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    FRAMES_DIR,
)

from src.preprocessing.video_processor import VideoProcessor


class DatasetBuilder:

    def __init__(self):

        self.processor = VideoProcessor(
            output_size=(224, 224)
        )

    def find_videos(self):

        videos = []

        for path in Path(RAW_DATA_DIR).rglob("*"):

            if path.is_file() and path.suffix.lower() == ".avi":
                videos.append(path)

        return sorted(videos)

    def process_all_videos(self):

        videos = self.find_videos()

        print(f"Found {len(videos)} videos.")

        if not videos:
            raise RuntimeError(
                f"No AVI files found inside {RAW_DATA_DIR}"
            )

        total_frames = 0

        for video_path in videos:

            relative_parent = video_path.parent.name

            output_dir = (
                Path(FRAMES_DIR) /
                relative_parent /
                video_path.stem
            )

            print(
                f"Processing: {video_path}"
            )

            count = self.processor.extract_frames(
                video_path,
                output_dir
            )

            total_frames += count

            print(
                f"  Saved {count} frames"
            )

        print(
            f"\nTotal frames extracted: {total_frames}"
        )

        return total_frames
