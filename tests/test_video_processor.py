"""Tests for the video_processor module."""

import os
import tempfile
from unittest.mock import patch

import pytest

from wwdcdigest.models import WWDCFrameSegment
from wwdcdigest.video_processor import DefaultVideoProcessor


@pytest.mark.anyio
async def test_default_video_processor():
    """Test the DefaultVideoProcessor."""
    processor = DefaultVideoProcessor()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real temporary subtitle file
        subtitle_path = os.path.join(temp_dir, "subtitle.vtt")
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

        video_path = "video.mp4"  # This will be mocked

        # Create test segments
        test_segments = [
            WWDCFrameSegment(
                timestamp="00:01:23.456",
                text="Test segment 1",
                image_path=os.path.join(temp_dir, "frame1.jpg"),
            ),
            WWDCFrameSegment(
                timestamp="00:02:34.567",
                text="Test segment 2",
                image_path=os.path.join(temp_dir, "frame2.jpg"),
            ),
        ]

        # Mock the extract_frames_from_video function
        with patch(
            "wwdcdigest.video_processor.extract_frames_from_video",
            return_value=test_segments,
        ) as mock_extract:
            # Call the method
            segments = await processor.extract_frames(
                video_path, subtitle_path, temp_dir, "jpg"
            )

            # Check that the function was called with the right arguments
            mock_extract.assert_called_once_with(
                video_path, subtitle_path, temp_dir, "jpg"
            )

            # Check the results
            assert segments == test_segments
            assert len(segments) == 2
            assert segments[0].timestamp == "00:01:23.456"
            assert segments[0].text == "Test segment 1"
            assert segments[1].timestamp == "00:02:34.567"
            assert segments[1].text == "Test segment 2"
