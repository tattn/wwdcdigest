"""Implementations of video processing components."""

import logging
from typing import Literal

from .models import WWDCFrameSegment
from .video import extract_frames_from_video

logger = logging.getLogger("wwdcdigest")


class DefaultVideoProcessor:
    """Default implementation of the VideoProcessor interface."""

    async def extract_frames(
        self,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        image_format: Literal["jpg", "png", "avif", "webp"] = "jpg",
    ) -> list[WWDCFrameSegment]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_dir: Directory to save extracted frames
            image_format: Format for the extracted images

        Returns:
            List of WWDCFrameSegment objects
        """
        logger.info(f"Extracting frames from {video_path} to {output_dir}")
        return extract_frames_from_video(
            video_path, subtitle_path, output_dir, image_format
        )
