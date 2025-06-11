"""Tests for the formatter module."""

import os
import tempfile

import pytest

from wwdcdigest.formatter import MarkdownFormatter
from wwdcdigest.models import WWDCDigest, WWDCFrameSegment


@pytest.mark.anyio
async def test_markdown_formatter():
    """Test the MarkdownFormatter."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        output_path = os.path.join(temp_dir, "test_digest.md")

        # Create test frames
        segments = [
            WWDCFrameSegment(
                timestamp="00:01:23.456",
                text="This is the first segment",
                image_path=os.path.join(temp_dir, "frame1.jpg"),
            ),
            WWDCFrameSegment(
                timestamp="00:02:34.567",
                text="This is the second segment",
                image_path=os.path.join(temp_dir, "frame2.jpg"),
            ),
        ]

        # Create test files to ensure the image paths exist
        for segment in segments:
            with open(segment.image_path, "w") as f:
                f.write("test image data")

        # Create digest
        digest = WWDCDigest(
            session_id="12345",
            title="Test Session",
            summary="This is a test summary",
            key_points=["Point 1", "Point 2", "Point 3"],
            segments=segments,
            source_url="https://example.com/test",
        )

        # Create formatter and format digest
        formatter = MarkdownFormatter()
        result_path = formatter.format_digest(digest, output_path)

        # Check that the file was created
        assert os.path.exists(result_path)
        assert result_path == output_path

        # Check the content of the file
        with open(result_path, encoding="utf-8") as f:
            content = f.read()

            # Check that all expected elements are in the content
            assert "# Test Session" in content
            assert "This is a test summary" in content
            assert "- Point 1" in content
            assert "- Point 2" in content
            assert "- Point 3" in content
            assert "This is the first segment" in content
            assert "This is the second segment" in content
            assert "frame1.jpg" in content
            assert "frame2.jpg" in content
            assert "00:01:23.456" in content
            assert "00:02:34.567" in content
