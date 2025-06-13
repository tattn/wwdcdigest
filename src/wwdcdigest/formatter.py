"""Implementations of digest formatting components."""

import logging
import os
from typing import TextIO

from .models import WWDCDigest
from .video import parse_webvtt_time

logger = logging.getLogger("wwdcdigest")


class MarkdownFormatter:
    """Implementation of DigestFormatter that creates markdown files."""

    def _insert_sample_codes_for_segment(
        self,
        f: TextIO,
        digest: WWDCDigest,
        start_time: int,
        end_time: int | None = None,
    ) -> None:
        """Insert sample codes that fall between start_time and end_time.

        Args:
            f: File handle to write to
            digest: The WWDCDigest object
            start_time: The start timestamp in seconds
            end_time: The end timestamp in seconds (optional)
        """
        if (
            not hasattr(digest.session, "sample_codes")
            or not digest.session.sample_codes
        ):
            return

        for sample_code in digest.session.sample_codes:
            if not sample_code.time:
                continue

            code_time = int(sample_code.time)

            # If end_time is None (last segment), include all codes after start_time
            # Otherwise, include codes that fall within the range [start_time, end_time)
            if (end_time is None and code_time >= start_time) or (
                end_time is not None and start_time <= code_time < end_time
            ):
                if sample_code.title:
                    f.write(f"#### {sample_code.title}\n\n")

                f.write("```\n")
                f.write(f"{sample_code.code}\n")
                f.write("```\n\n")

    def format_digest(self, digest: WWDCDigest, output_path: str) -> str:
        """Format a digest as a markdown file.

        Args:
            digest: The WWDCDigest object to format
            output_path: Path to save the formatted output

        Returns:
            Path to the created markdown file
        """
        logger.info(f"Creating markdown file at {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"# {digest.session.title}\n\n")

            # Write source URL if available
            if digest.source_url:
                f.write(f"Source: [{digest.source_url}]({digest.source_url})\n\n")

            # Write summary
            f.write("## Summary\n\n")
            f.write(f"{digest.summary}\n\n")

            # Write key points if any
            if digest.key_points:
                f.write("## Key Points\n\n")
                for point in digest.key_points:
                    f.write(f"- {point}\n")
                f.write("\n")

            # Write transcript with images
            f.write("## Transcript with Video Frames\n\n")

            # Process segments
            for i, segment in enumerate(digest.segments):
                # Make image path relative to the markdown file
                image_rel_path = os.path.relpath(
                    segment.image_path, os.path.dirname(output_path)
                )

                # Calculate timestamp in seconds
                timestamp_seconds = int(parse_webvtt_time(segment.timestamp))

                # Calculate end time if this isn't the last segment
                end_time = None
                if i < len(digest.segments) - 1:
                    end_time = int(parse_webvtt_time(digest.segments[i + 1].timestamp))

                # Write timestamp as heading with link to the specific time in the video
                if digest.source_url:
                    timestamp_url = f"{digest.source_url}?time={timestamp_seconds}"
                    f.write(f"### [{segment.timestamp}]({timestamp_url})\n\n")
                else:
                    f.write(f"### {segment.timestamp}\n\n")

                # Write text
                f.write(f"{segment.text}\n\n")

                # Write image
                f.write(f"![Frame at {segment.timestamp}]({image_rel_path})\n\n")

                # Insert sample codes for this segment
                self._insert_sample_codes_for_segment(
                    f, digest, timestamp_seconds, end_time
                )

                # Add separator
                f.write("---\n\n")

        return output_path
