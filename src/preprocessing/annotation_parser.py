from pathlib import Path


class AnnotationParser:

    def __init__(self, annotation_root):
        self.annotation_root = Path(annotation_root)

    def find_annotation(self, video_path):

        video_path = Path(video_path)

        candidates = [
            video_path.with_suffix(".txt"),
            video_path.parent.parent / "Annotation_files" /
            f"{video_path.stem}.txt",
            video_path.parent.parent /
            "Annotation_files" /
            f"{video_path.stem}.txt",
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

        matches = list(
            self.annotation_root.rglob(
                f"{video_path.stem}.txt"
            )
        )

        if matches:
            return matches[0]

        return None

    def parse_fall_interval(self, annotation_file):

        if annotation_file is None:
            return None

        annotation_file = Path(annotation_file)

        try:
            lines = [
                line.strip()
                for line in annotation_file.read_text(
                    errors="ignore"
                ).splitlines()
                if line.strip()
            ]
        except Exception:
            return None

        if len(lines) < 2:
            return None

        values = []

        for line in lines[:2]:

            try:
                value = int(float(line.split()[0]))
                values.append(value)
            except (ValueError, IndexError):
                return None

        if len(values) != 2:
            return None

        start_frame, end_frame = values

        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        return start_frame, end_frame

    def is_fall_frame(self, frame_index, interval):

        if interval is None:
            return False

        start_frame, end_frame = interval

        return start_frame <= frame_index <= end_frame
