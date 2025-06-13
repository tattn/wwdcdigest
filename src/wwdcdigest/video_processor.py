"""Implementations of video processing components."""

import logging

from .models import ImageOptions, WWDCFrameSegment
from .video import extract_frames_from_video, load_segments_from_frames_dir

logger = logging.getLogger("wwdcdigest")


class DefaultVideoProcessor:
    """Default implementation of the VideoProcessor interface."""

    async def extract_frames(
        self,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        image_options: ImageOptions,
    ) -> list[WWDCFrameSegment]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_dir: Directory to save extracted frames
            image_options: Options for image extraction and formatting

        Returns:
            List of WWDCFrameSegment objects
        """
        logger.info(f"Extracting frames from {video_path} to {output_dir}")
        return extract_frames_from_video(
            video_path, subtitle_path, output_dir, image_options
        )

    async def load_segments_from_frames(
        self,
        frames_dir: str,
    ) -> list[WWDCFrameSegment]:
        """Load existing frame segments from a directory.

        Args:
            frames_dir: Directory containing frame images

        Returns:
            List of WWDCFrameSegment objects
        """
        return load_segments_from_frames_dir(frames_dir)
