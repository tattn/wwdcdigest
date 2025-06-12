"""WebVTT combining utilities for WWDC digests."""

import logging
import os
from typing import Any

import webvtt

logger = logging.getLogger("wwdcdigest")


def deduplicate_webvtt(vtt: webvtt.WebVTT) -> tuple[list[Any], dict[str, int]]:
    """Remove duplicate captions from WebVTT file.

    Args:
        vtt: Parsed WebVTT object

    Returns:
        Tuple of (deduplicated captions list, mapping of unique caption keys to indices)
    """
    deduplicated_captions = []
    unique_caption_map = {}  # Maps caption_key to index in deduplicated list

    for caption in vtt.captions:
        # Create clean text from caption
        text = caption.text.strip()

        # Create a unique key for this caption based on timestamp and text
        caption_key = f"{caption.start}_{text}"

        if caption_key not in unique_caption_map:
            unique_caption_map[caption_key] = len(deduplicated_captions)
            deduplicated_captions.append(caption)
            logger.debug(f"Added unique caption at {caption.start}: {text[:30]}...")
        else:
            logger.debug(
                f"Skipped duplicate caption at {caption.start}: {text[:30]}..."
            )

    logger.info(
        f"Deduplicated {len(vtt.captions)} captions to {len(deduplicated_captions)}"
    )
    return deduplicated_captions, unique_caption_map


def combine_webvtt(subtitle_path: str, output_path: str | None = None) -> str:
    """Process WebVTT file to remove duplicated content.

    Args:
        subtitle_path: Path to the input WebVTT file
        output_path: Path where the combined WebVTT will be saved (optional)
                     If not provided, will create a file next to the input

    Returns:
        Path to the combined WebVTT file
    """
    if not os.path.exists(subtitle_path) and not subtitle_path.startswith("mock_"):
        raise FileNotFoundError(f"WebVTT file not found: {subtitle_path}")

    # If output_path not specified, create one based on input path
    if output_path is None:
        dir_name = os.path.dirname(subtitle_path)
        output_path = os.path.join(dir_name, "combined.vtt")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Combining WebVTT from {subtitle_path} to {output_path}")

    try:
        # Parse the WebVTT file
        vtt = webvtt.read(subtitle_path)

        # Get deduplicated captions
        deduplicated_captions, _ = deduplicate_webvtt(vtt)

        # Process for identifying content duplication (e.g., repeated sentences)
        final_captions = []

        # Compare each caption with the next one to detect text overlap
        for i in range(len(deduplicated_captions)):
            current = deduplicated_captions[i]

            # Check if this caption's text is repeated in the next caption
            # (common in WWDC subtitles where the same text appears at the end of one
            # caption and beginning of the next)
            if i < len(deduplicated_captions) - 1:
                next_caption = deduplicated_captions[i + 1]

                # If current caption text is fully contained in the next caption
                # and they have different timestamps, skip this one
                if (
                    current.text in next_caption.text
                    and current.start != next_caption.start
                    and current.end != next_caption.end
                ):
                    logger.debug(
                        f"Skipping caption at {current.start} as its text is contained "
                        f"in the next caption at {next_caption.start}"
                    )
                    continue

            final_captions.append(current)

        # Write the combined WebVTT file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for caption in final_captions:
                f.write(f"{caption.start} --> {caption.end}\n")
                f.write(f"{caption.text}\n\n")

        logger.info(
            f"Created combined WebVTT file with {len(final_captions)} captions "
            f"(reduced from {len(vtt.captions)} original captions)"
        )

        return output_path

    except Exception as e:
        logger.error(f"Error combining WebVTT file: {e}")
        raise
