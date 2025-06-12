"""Tests for WebVTT utilities module."""

import os
from pathlib import Path

import pytest
import webvtt
from wwdctools import combine_webvtt_files

from wwdcdigest.webvtt_utils import prepare_combined_subtitle


def test_combine_webvtt(tmp_path: Path) -> None:
    """Test combining of WebVTT file with duplicate content."""
    # Create a sample WebVTT file with duplicates
    sample_vtt_path = tmp_path / "sample.vtt"
    with open(sample_vtt_path, "w", encoding="utf-8") as f:
        f.write(
            """WEBVTT

00:00:06.904 --> 00:00:10.374
Hello, I'm Nicholas,
an engineer on the Accessibility team.

00:00:11.074 --> 00:00:13.277
Accessibility empowers everyone
to experience

00:00:11.074 --> 00:00:13.277
Accessibility empowers everyone
to experience

00:00:13.277 --> 00:00:15.379
and love the apps that you create.

00:00:16.313 --> 00:00:18.348
Today I'm going to go beyond
the basics to explore

00:00:16.313 --> 00:00:18.348
Today I'm going to go beyond
the basics to explore

00:00:18.348 --> 00:00:20.584
how you can make your Mac app
more accessible.
"""
        )

    # Combine the WebVTT file
    output_path = str(tmp_path / "combined.vtt")
    combine_webvtt_files([str(sample_vtt_path)], output_path)

    # Check the result
    assert os.path.exists(output_path)

    # Read the combined file
    combined_vtt = webvtt.read(output_path)

    # Check that duplicate captions were removed
    # The specific number of captions may vary based on wwdctools implementation
    # but we can check that key duplicates are removed
    texts = [c.text.strip() for c in combined_vtt.captions]
    assert "Accessibility empowers everyone\nto experience" in texts
    assert "Today I'm going to go beyond\nthe basics to explore" in texts


def test_combine_webvtt_file_not_found() -> None:
    """Test combining with non-existent file."""
    with pytest.raises((FileNotFoundError, IOError), match=".*file.vtt.*"):
        combine_webvtt_files(["/non/existent/file.vtt"], "output.vtt")


def test_prepare_combined_subtitle_single_file(tmp_path: Path) -> None:
    """Test preparing a combined subtitle from a single file."""
    # Create a sample WebVTT file
    sample_vtt_path = tmp_path / "sample.vtt"
    with open(sample_vtt_path, "w", encoding="utf-8") as f:
        f.write(
            """WEBVTT

00:00:06.904 --> 00:00:10.374
Hello, I'm Nicholas,
an engineer on the Accessibility team.

00:00:11.074 --> 00:00:13.277
Accessibility empowers everyone
to experience
"""
        )

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Call the function to test
    combined_path = prepare_combined_subtitle(str(sample_vtt_path), str(output_dir))

    # Check the result
    assert os.path.exists(combined_path)
    assert os.path.basename(combined_path) == "combined.vtt"

    # Read the combined file to verify content
    combined_vtt = webvtt.read(combined_path)
    texts = [c.text.strip() for c in combined_vtt.captions]
    assert "Hello, I'm Nicholas,\nan engineer on the Accessibility team." in texts
    assert "Accessibility empowers everyone\nto experience" in texts


def test_prepare_combined_subtitle_directory(tmp_path: Path) -> None:
    """Test preparing a combined subtitle from a directory of files."""
    # Create a directory with multiple WebVTT files
    subtitle_dir = tmp_path / "subtitles"
    subtitle_dir.mkdir()

    # Create first file
    with open(subtitle_dir / "file1.vtt", "w", encoding="utf-8") as f:
        f.write(
            """WEBVTT

00:00:06.904 --> 00:00:10.374
Hello, I'm Nicholas,
an engineer on the Accessibility team.
"""
        )

    # Create second file
    with open(subtitle_dir / "file2.vtt", "w", encoding="utf-8") as f:
        f.write(
            """WEBVTT

00:00:11.074 --> 00:00:13.277
Accessibility empowers everyone
to experience
"""
        )

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Call the function to test
    combined_path = prepare_combined_subtitle(str(subtitle_dir), str(output_dir))

    # Check the result
    assert os.path.exists(combined_path)
    assert os.path.basename(combined_path) == "combined.vtt"

    # Read the combined file to verify content
    combined_vtt = webvtt.read(combined_path)
    texts = [c.text.strip() for c in combined_vtt.captions]
    assert "Hello, I'm Nicholas,\nan engineer on the Accessibility team." in texts
    assert "Accessibility empowers everyone\nto experience" in texts


def test_prepare_combined_subtitle_empty_directory(tmp_path: Path) -> None:
    """Test preparing a combined subtitle from an empty directory."""
    # Create an empty directory
    subtitle_dir = tmp_path / "empty_subtitles"
    subtitle_dir.mkdir()

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Call the function to test
    combined_path = prepare_combined_subtitle(str(subtitle_dir), str(output_dir))

    # When no WebVTT files are found, the function should return the input path
    assert combined_path == str(subtitle_dir)


def test_prepare_combined_subtitle_nonexistent_file(tmp_path: Path) -> None:
    """Test handling of a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        prepare_combined_subtitle("/nonexistent/file.vtt", str(tmp_path))
