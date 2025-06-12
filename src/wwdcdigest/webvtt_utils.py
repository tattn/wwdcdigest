"""WebVTT processing utilities for WWDC digests."""

import logging
import os
import re

import webvtt
from wwdctools import combine_webvtt_files

logger = logging.getLogger("wwdcdigest")

# Constants for WebVTT parsing
HOURS_MINUTES_SECONDS = 3
MINUTES_SECONDS = 2


def parse_webvtt_time(time_str: str) -> float:
    """Convert WebVTT timestamp to seconds.

    Args:
        time_str: WebVTT timestamp (e.g., '00:01:23.456')

    Returns:
        Time in seconds as a float
    """
    parts = time_str.split(":")
    if len(parts) == HOURS_MINUTES_SECONDS:  # HH:MM:SS.sss
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == MINUTES_SECONDS:  # MM:SS.sss
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(time_str)


def prepare_combined_subtitle(subtitle_path: str, output_dir: str) -> str:
    """Prepare a combined subtitle file.

    Args:
        subtitle_path: Path to the subtitle file or directory
        output_dir: Output directory for the combined subtitle

    Returns:
        Path to the combined subtitle file
    """
    combined_subtitle_path = os.path.join(output_dir, "combined.vtt")

    # If subtitle_path is a single file, use it directly with combine_webvtt_files
    if os.path.isfile(subtitle_path):
        combine_webvtt_files([subtitle_path], combined_subtitle_path)
    else:
        # If it's a directory, find all WebVTT files and combine them
        webvtt_files = [
            os.path.join(subtitle_path, f)
            for f in os.listdir(subtitle_path)
            if f.endswith((".webvtt", ".vtt"))
        ]
        if webvtt_files:
            combine_webvtt_files(webvtt_files, combined_subtitle_path)
        else:
            # No WebVTT files found, use the subtitle_path directly
            combined_subtitle_path = subtitle_path

    return combined_subtitle_path


def prepare_subtitle_path(subtitle_path: str, output_dir: str) -> str:
    """Prepare subtitle path, combining WebVTT files if needed.

    Args:
        subtitle_path: Path to the subtitle file or directory
        output_dir: Output directory for combined subtitle

    Returns:
        Path to the prepared subtitle file
    """
    combined_subtitle_path = os.path.join(output_dir, "combined.vtt")

    # If subtitle_path is a directory with multiple .webvtt files, combine them
    if os.path.isdir(subtitle_path):
        logger.debug(f"Combining WebVTT files from {subtitle_path}")
        with open(combined_subtitle_path, "w", encoding="utf-8") as outfile:
            outfile.write("WEBVTT\n\n")

            # Sort by sequence number (sequence_1.webvtt, sequence_2.webvtt, ...)
            webvtt_files = [
                f for f in os.listdir(subtitle_path) if f.endswith(".webvtt")
            ]

            def get_sequence_number(filename: str) -> int:
                match = re.search(r"sequence_(\d+)", filename)
                return int(match.group(1)) if match else 0

            sorted_files = sorted(webvtt_files, key=get_sequence_number)

            # Track unique captions to avoid duplicates
            unique_captions = set()

            for filename in sorted_files:
                file_path = os.path.join(subtitle_path, filename)
                with open(file_path, encoding="utf-8") as infile:
                    content = infile.read()
                    # Remove header if present
                    if content.startswith("WEBVTT"):
                        content = re.sub(
                            r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL
                        )

                    # Parse and deduplicate captions
                    captions = []
                    current_caption = []

                    for line in content.splitlines():
                        if line.strip() == "":
                            if current_caption:
                                caption_text = "\n".join(current_caption)
                                if caption_text not in unique_captions:
                                    unique_captions.add(caption_text)
                                    captions.append(caption_text)
                                current_caption = []
                        else:
                            current_caption.append(line)

                    # Add any remaining caption
                    if current_caption:
                        caption_text = "\n".join(current_caption)
                        if caption_text not in unique_captions:
                            unique_captions.add(caption_text)
                            captions.append(caption_text)

                    # Write unique captions to the output file
                    for caption in captions:
                        outfile.write(caption)
                        outfile.write("\n\n")

        logger.debug(
            f"Created combined WebVTT file with {len(unique_captions)} unique captions"
        )
        return combined_subtitle_path

    return subtitle_path


def read_webvtt(subtitle_path: str) -> webvtt.WebVTT:
    """Read WebVTT file and return parsed content.

    Args:
        subtitle_path: Path to the WebVTT file

    Returns:
        Parsed WebVTT content
    """
    return webvtt.read(subtitle_path)
