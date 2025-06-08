"""Tests for video processing utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from wwdcdigest.video import (
    SIMILARITY_THRESHOLD,
    _prepare_subtitle_path,
    compare_images,
    extract_frames_from_video,
    parse_webvtt_time,
)


def test_parse_webvtt_time():
    """Test parsing WebVTT timestamps."""
    # Test hours, minutes, seconds format
    assert parse_webvtt_time("01:23:45.678") == 5025.678
    # Test minutes, seconds format
    assert parse_webvtt_time("12:34.567") == 754.567
    # Test seconds only
    assert parse_webvtt_time("12.345") == 12.345


@pytest.mark.anyio
async def test_compare_images():
    """Test image comparison."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two test images (black squares)
        import cv2
        import numpy as np

        # Create identical images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save images
        img1_path = os.path.join(temp_dir, "img1.jpg")
        img2_path = os.path.join(temp_dir, "img2.jpg")
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)

        # Compare identical images
        similarity = compare_images(img1_path, img2_path)
        assert similarity >= SIMILARITY_THRESHOLD

        # Create a different image (white square)
        img3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img3_path = os.path.join(temp_dir, "img3.jpg")
        cv2.imwrite(img3_path, img3)

        # Compare different images
        similarity = compare_images(img1_path, img3_path)
        assert similarity < SIMILARITY_THRESHOLD


@pytest.mark.anyio
async def test_extract_frames_merges_similar_frames():
    """Test that similar consecutive frames are merged."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock video and subtitle data
        video_path = "mock_video.mp4"
        subtitle_path = "mock_subtitle.vtt"
        output_dir = temp_dir

        # Create a mock for cv2.VideoCapture
        mock_video = MagicMock()
        mock_video.isOpened.return_value = True

        # Define a side effect function with explicit typing
        def get_mock(prop: int) -> float:
            return 30.0 if prop == cv2.CAP_PROP_FPS else 300.0

        mock_video.get.side_effect = get_mock

        # Create similar mock frames (black images)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a different frame (white image)
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Set up read to return our frames
        mock_video.read.side_effect = [
            (True, frame1),
            (True, frame2),  # Similar to frame1, should be merged
            (True, frame3),  # Different, should be a new segment
        ]

        # Mock WebVTT data
        mock_captions = [
            MagicMock(start="00:00:10.000", text="Caption 1"),
            MagicMock(
                start="00:00:15.000", text="Caption 2"
            ),  # Should be merged with Caption 1
            MagicMock(
                start="00:00:20.000", text="Caption 3"
            ),  # Different frame, should be separate
        ]
        mock_vtt = MagicMock()
        mock_vtt.__iter__.return_value = mock_captions

        # Patch dependencies
        with (
            patch("wwdcdigest.video.cv2.VideoCapture", return_value=mock_video),
            patch("wwdcdigest.video.webvtt.read", return_value=mock_vtt),
            patch("wwdcdigest.video.compare_images") as mock_compare,
        ):
            # Set up compare_images to return high similarity for the first two frames
            # and low similarity for the third
            mock_compare.side_effect = [0.98, 0.3]

            # Call the function
            segments = extract_frames_from_video(video_path, subtitle_path, output_dir)

            # Check that we got 2 segments (3 frames with 2 similar ones merged)
            assert len(segments) == 2

            # Check that the first segment has merged text
            assert "Caption 1" in segments[0].text
            assert "Caption 2" in segments[0].text

            # Check that the third caption is in the second segment
            assert segments[1].text == "Caption 3"


@pytest.mark.anyio
async def test_prepare_subtitle_path():
    """Test preparing subtitle path with multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock subtitle directory with multiple .webvtt files
        subtitle_dir = Path(temp_dir) / "subtitles"
        subtitle_dir.mkdir()

        # Create multiple subtitle files
        (subtitle_dir / "part1.webvtt").write_text(
            "WEBVTT\n\n1\n00:00:10.000 --> 00:00:15.000\nPart 1"
        )
        (subtitle_dir / "part2.webvtt").write_text(
            "WEBVTT\n\n2\n00:00:20.000 --> 00:00:25.000\nPart 2"
        )

        # Test combining subtitles
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        combined_path = _prepare_subtitle_path(str(subtitle_dir), str(output_dir))

        # Check that the combined file exists and contains both parts
        assert os.path.exists(combined_path)
        with open(combined_path, encoding="utf-8") as f:
            content = f.read()
            assert "Part 1" in content
            assert "Part 2" in content
