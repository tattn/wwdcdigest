"""Tests for WebVTT combining functionality."""

import os
from pathlib import Path

import pytest
import webvtt

from wwdcdigest.webvtt_combiner import combine_webvtt, deduplicate_webvtt


def test_deduplicate_webvtt(tmp_path: Path) -> None:
    """Test deduplication of WebVTT captions."""
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
"""
        )

    # Read the WebVTT file
    vtt = webvtt.read(str(sample_vtt_path))

    # Deduplicate captions
    deduplicated, unique_map = deduplicate_webvtt(vtt)

    # Check results
    assert len(vtt.captions) == 4
    assert len(deduplicated) == 3
    assert len(unique_map) == 3

    # Check that the duplicate caption was removed
    texts = [c.text.strip() for c in deduplicated]
    assert "Accessibility empowers everyone\nto experience" in texts
    assert texts.count("Accessibility empowers everyone\nto experience") == 1


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
    result_path = combine_webvtt(str(sample_vtt_path), output_path)

    # Check the result
    assert result_path == output_path
    assert os.path.exists(result_path)

    # Read the combined file
    combined_vtt = webvtt.read(result_path)

    # Check the number of captions (should be 5 instead of 7)
    assert len(combined_vtt.captions) == 5

    # Check that the duplicate captions were removed
    texts = [c.text.strip() for c in combined_vtt.captions]
    assert "Accessibility empowers everyone\nto experience" in texts
    assert texts.count("Accessibility empowers everyone\nto experience") == 1
    assert "Today I'm going to go beyond\nthe basics to explore" in texts
    assert texts.count("Today I'm going to go beyond\nthe basics to explore") == 1


def test_combine_webvtt_default_output(tmp_path: Path) -> None:
    """Test combining of WebVTT with default output path."""
    # Create a sample WebVTT file
    sample_vtt_path = tmp_path / "sample.vtt"
    with open(sample_vtt_path, "w", encoding="utf-8") as f:
        f.write(
            """WEBVTT

00:00:06.904 --> 00:00:10.374
Hello, I'm Nicholas,
an engineer on the Accessibility team.
"""
        )

    # Combine the WebVTT file with default output
    result_path = combine_webvtt(str(sample_vtt_path))

    # Check the result
    expected_path = str(tmp_path / "combined.vtt")
    assert result_path == expected_path
    assert os.path.exists(result_path)


def test_combine_webvtt_file_not_found() -> None:
    """Test combining with non-existent file."""
    with pytest.raises(FileNotFoundError):
        combine_webvtt("/non/existent/file.vtt")
