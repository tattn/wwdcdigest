"""Tests for video processing utilities."""

import os
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from wwdcdigest.models import ImageOptions
from wwdcdigest.video import (
    SIMILARITY_THRESHOLD,
    compare_images,
    delete_unused_image_files,
    extract_frames_from_video,
)
from wwdcdigest.webvtt_utils import parse_webvtt_time, prepare_subtitle_path


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
        # Create a real temporary subtitle file
        subtitle_path = os.path.join(temp_dir, "subtitle.vtt")
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

        video_path = "video.mp4"  # This will be mocked
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
            (False, None),  # End of video
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

        # Create a mock for _save_frame_image that does nothing
        def mock_save_frame(
            frame: np.ndarray, path: str, image_options: "ImageOptions"
        ) -> None:
            pass

        # Patch dependencies
        with (
            patch("wwdcdigest.video.cv2.VideoCapture", return_value=mock_video),
            patch("wwdctools.combine_webvtt_files", return_value=None),
            patch("wwdcdigest.video.webvtt.read", return_value=mock_vtt),
            patch("wwdcdigest.video.compare_images") as mock_compare,
            patch("wwdcdigest.video._save_frame_image", side_effect=mock_save_frame),
            # Add a patch for vtt.captions to ensure it's properly accessible
            patch.object(mock_vtt, "captions", mock_captions),
        ):
            # Set up compare_images to return high similarity for the first two frames
            # and low similarity for the third
            mock_compare.side_effect = [0.98, 0.3]

            # Call the function with ImageOptions
            from wwdcdigest.models import ImageOptions

            image_options = ImageOptions(format="jpg")
            segments = extract_frames_from_video(
                video_path, subtitle_path, output_dir, image_options
            )

            # Check that we got 2 segments (3 frames with 2 similar ones merged)
            assert len(segments) == 2

            # Check that the first segment has merged text
            assert "Caption 1" in segments[0].text
            assert "Caption 2" in segments[0].text

            # Check that the third caption is in the second segment
            assert segments[1].text == "Caption 3"


@pytest.mark.anyio
async def testprepare_subtitle_path():
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

        combined_path = prepare_subtitle_path(str(subtitle_dir), str(output_dir))

        # Check that the combined file exists and contains both parts
        assert os.path.exists(combined_path)
        with open(combined_path, encoding="utf-8") as f:
            content = f.read()
            assert "Part 1" in content
            assert "Part 2" in content


@pytest.mark.anyio
async def testprepare_subtitle_path_deduplicates_captions():
    """Test that duplicate captions are removed when combining subtitle files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock subtitle directory with multiple .webvtt files
        subtitle_dir = Path(temp_dir) / "subtitles"
        subtitle_dir.mkdir()

        # Create subtitle files with duplicate content
        (subtitle_dir / "sequence_1.webvtt").write_text(
            "WEBVTT\n\n"
            "00:00:06.904 --> 00:00:10.374 align:center line:79%\n"
            "Hello, I'm Nicholas,\n"
            "an engineer on the Accessibility team.\n\n"
            "00:00:11.074 --> 00:00:13.277 align:center line:79%\n"
            "Accessibility empowers everyone\n"
            "to experience\n\n"
        )

        (subtitle_dir / "sequence_2.webvtt").write_text(
            "WEBVTT\n\n"
            "00:00:11.074 --> 00:00:13.277 align:center line:79%\n"
            "Accessibility empowers everyone\n"
            "to experience\n\n"
            "00:00:13.277 --> 00:00:15.379 align:center line:81%\n"
            "and love the apps that you create.\n\n"
        )

        (subtitle_dir / "sequence_3.webvtt").write_text(
            "WEBVTT\n\n"
            "00:00:16.313 --> 00:00:18.348 align:center line:79%\n"
            "Today I'm going to go beyond\n"
            "the basics to explore\n\n"
            "00:00:16.313 --> 00:00:18.348 align:center line:79%\n"
            "Today I'm going to go beyond\n"
            "the basics to explore\n\n"
            "00:00:18.348 --> 00:00:20.584 align:center line:79%\n"
            "how you can make your Mac app\n"
            "more accessible.\n\n"
        )

        # Test combining subtitles
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        combined_path = prepare_subtitle_path(str(subtitle_dir), str(output_dir))

        # Check that the combined file exists
        assert os.path.exists(combined_path)

        # Read the combined file
        with open(combined_path, encoding="utf-8") as f:
            content = f.read()

        # Count the number of occurrences of each unique caption
        assert (
            content.count(
                "00:00:11.074 --> 00:00:13.277 align:center line:79%\n"
                "Accessibility empowers everyone\n"
                "to experience"
            )
            == 1
        )

        assert (
            content.count(
                "00:00:16.313 --> 00:00:18.348 align:center line:79%\n"
                "Today I'm going to go beyond\n"
                "the basics to explore"
            )
            == 1
        )

        # Check that all expected captions are present exactly once
        expected_captions = [
            "Hello, I'm Nicholas,\nan engineer on the Accessibility team.",
            "Accessibility empowers everyone\nto experience",
            "and love the apps that you create.",
            "Today I'm going to go beyond\nthe basics to explore",
            "how you can make your Mac app\nmore accessible.",
        ]

        for caption in expected_captions:
            assert caption in content, f"Caption not found: {caption}"


@pytest.mark.anyio
async def test_delete_unused_image_files():
    """Test deleting unused image files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image files
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
            # Create an empty file
            with open(file_path, "w") as f:
                f.write("")
            test_files.append(file_path)

        # Make sure files exist
        for file_path in test_files:
            assert os.path.exists(file_path)

        # Delete the files
        delete_unused_image_files(test_files)

        # Verify files are deleted
        for file_path in test_files:
            assert not os.path.exists(file_path)

        # Test with non-existent files (should not raise exception)
        non_existent_file = os.path.join(temp_dir, "non_existent.jpg")
        delete_unused_image_files([non_existent_file])


@pytest.mark.anyio
async def test_image_format_options():
    """Test that different image formats are supported."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a real temporary subtitle file
        subtitle_path = os.path.join(temp_dir, "subtitle.vtt")
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

        video_path = "video.mp4"  # This will be mocked
        output_dir = temp_dir

        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test different image formats
        formats: list[Literal["jpg", "png", "avif", "webp"]] = [
            "jpg",
            "png",
            "avif",
            "webp",
        ]

        for fmt in formats:
            # Create a mock for cv2.VideoCapture for each format
            mock_video = MagicMock()
            mock_video.isOpened.return_value = True
            mock_video.get.return_value = 30.0  # fps

            # Set up read to return our frame and then False for end of video
            mock_video.read.side_effect = [(True, frame), (False, None)]

            # Mock WebVTT data
            mock_caption = MagicMock(start="00:00:10.000", text="Test Caption")
            mock_vtt = MagicMock()
            mock_vtt.__iter__.return_value = [mock_caption]

            # Create a mock for _save_frame_image that does nothing
            def mock_save_frame(
                frame: np.ndarray, path: str, image_options: "ImageOptions"
            ) -> None:
                pass

            # Patch dependencies
            with (
                patch("wwdcdigest.video.cv2.VideoCapture", return_value=mock_video),
                patch("wwdctools.combine_webvtt_files", return_value=None),
                patch("wwdcdigest.video.webvtt.read", return_value=mock_vtt),
                patch("wwdcdigest.video.compare_images", return_value=0.5),
                patch(
                    "wwdcdigest.video._save_frame_image", side_effect=mock_save_frame
                ) as mock_save,
                # Add a patch for vtt.captions to ensure it's properly accessible
                patch.object(mock_vtt, "captions", [mock_caption]),
            ):
                # Call the function with the current format using ImageOptions
                from wwdcdigest.models import ImageOptions

                image_options = ImageOptions(format=fmt)
                segments = extract_frames_from_video(
                    video_path, subtitle_path, output_dir, image_options
                )

                # Check that the save function was called with the right format
                # We need to adapt this assertion to match the new signature
                mock_save.assert_called_once()
                args, _ = mock_save.call_args
                assert args[0] is frame  # Check frame is the same
                assert args[1] == os.path.join(
                    output_dir, f"frame_0000.{fmt}"
                )  # Check path
                assert args[2].format == fmt  # Check format in ImageOptions

                # Check that we got the segment
                assert len(segments) == 1
                assert segments[0].text == "Test Caption"
                assert segments[0].image_path.endswith(f".{fmt}")
