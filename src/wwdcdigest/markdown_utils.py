"""Functions for creating markdown documents from WWDC digests."""

import logging
import os

from .models import WWDCDigest

logger = logging.getLogger("wwdcdigest")


def create_markdown(digest: WWDCDigest, output_path: str) -> str:
    """Create a markdown file from the digest.

    Args:
        digest: WWDCDigest object
        output_path: Path to save the markdown file

    Returns:
        Path to the created markdown file
    """
    logger.info(f"Creating markdown file at {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# {digest.title}\n\n")
        f.write(f"WWDC Session: {digest.session_id}\n\n")

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

            # Write timestamp as heading
            f.write(f"### {segment.timestamp}\n\n")

            # Write text
            f.write(f"{segment.text}\n\n")

            # Write image
            f.write(f"![Frame at {segment.timestamp}]({image_rel_path})\n\n")

            # Add separator
            f.write("---\n\n")

    return output_path
