"""Implementations of digest formatting components."""

import logging
import os

from .models import WWDCDigest
from .video import parse_webvtt_time

logger = logging.getLogger("wwdcdigest")


class MarkdownFormatter:
    """Implementation of DigestFormatter that creates markdown files."""

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
            f.write(f"# {digest.title}\n\n")

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

            for segment in digest.segments:
                # Make image path relative to the markdown file
                image_rel_path = os.path.relpath(
                    segment.image_path, os.path.dirname(output_path)
                )

                # Calculate timestamp in seconds
                timestamp_seconds = int(parse_webvtt_time(segment.timestamp))

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

                # Add separator
                f.write("---\n\n")

        return output_path
