"""Video processing utilities for WWDC digests."""

import logging
import os
import re
from typing import Literal

import cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401
import webvtt
from PIL import Image

from .models import WWDCFrameSegment

logger = logging.getLogger("wwdcdigest")

# Constants for WebVTT parsing
HOURS_MINUTES_SECONDS = 3
MINUTES_SECONDS = 2

# Constants for image comparison
# Higher value means images need to be more similar to be merged
SIMILARITY_THRESHOLD = 0.95


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


def extract_frames_from_video(
    video_path: str,
    subtitle_path: str,
    output_dir: str,
    image_format: Literal["jpg", "png", "avif", "webp"] = "jpg",
) -> list[WWDCFrameSegment]:
    """Extract frames from video at subtitle timestamps.

    Args:
        video_path: Path to the video file
        subtitle_path: Path to the WebVTT subtitle file
        output_dir: Directory to save extracted frames
        image_format: Format to save the image files (jpg, png, avif, webp)

    Returns:
        List of WWDCFrameSegment objects with timestamp, text and image path
    """
    logger.info(f"Extracting frames from {video_path} using {subtitle_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    logger.debug(f"Video: {fps} fps, {duration:.2f} seconds")

    # Parse WebVTT
    segments = []
    subtitle_path = _prepare_subtitle_path(subtitle_path, output_dir)

    # Parse WebVTT file
    try:
        # Parse WebVTT
        vtt = webvtt.read(subtitle_path)
        raw_segments = []

        # Track unique captions to deduplicate identical entries
        unique_captions = {}

        # First pass: extract all frames without merging
        for i, caption in enumerate(vtt.captions):
            start_time = parse_webvtt_time(caption.start)

            # Create clean text from caption
            text = caption.text.strip()

            # Skip if this is a duplicate caption with the same timestamp and text
            caption_key = f"{caption.start}_{text}"
            if caption_key in unique_captions:
                logger.debug(
                    f"Skipping duplicate caption at {caption.start}: {text[:30]}..."
                )
                continue

            unique_captions[caption_key] = i

            # Seek to timestamp position
            video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

            # Read frame
            success, frame = video.read()
            if success:
                # Save frame as image with the specified format
                image_filename = f"frame_{i:04d}.{image_format}"
                image_path = os.path.join(output_dir, image_filename)
                _save_frame_image(frame, image_path, image_format)

                # Create segment
                segment = WWDCFrameSegment(
                    timestamp=caption.start, text=text, image_path=image_path
                )
                raw_segments.append(segment)
                logger.debug(f"Extracted frame at {caption.start}: {text[:30]}...")
            else:
                logger.warning(f"Failed to extract frame at {caption.start}")

        # Second pass: merge similar consecutive frames
        merged_segments, unused_image_files = _process_raw_segments(raw_segments)

        # Delete unused image files
        if unused_image_files:
            delete_unused_image_files(unused_image_files)
            logger.info(f"Deleted {len(unused_image_files)} unused image files")

        segments = merged_segments
        logger.info(
            f"Extracted {len(raw_segments)} frames, "
            f"merged to {len(segments)} unique segments"
        )

    except Exception as e:
        logger.error(f"Error parsing WebVTT file: {e}")

    # Release video
    video.release()

    return segments


def _process_raw_segments(
    raw_segments: list[WWDCFrameSegment],
) -> tuple[list[WWDCFrameSegment], list[str]]:
    """Process raw segments by merging similar consecutive frames.

    Args:
        raw_segments: Raw frame segments extracted from video

    Returns:
        Tuple of (merged segments, unused image files)
    """
    if not raw_segments:
        return [], []

    merged_segments = [raw_segments[0]]
    unused_image_files = []  # Track image files that will no longer be used

    for i in range(1, len(raw_segments)):
        current_segment = raw_segments[i]
        prev_segment = merged_segments[-1]

        # Compare current frame with the last merged frame
        similarity = compare_images(prev_segment.image_path, current_segment.image_path)

        if similarity >= SIMILARITY_THRESHOLD:
            # Frames are similar, merge by keeping the first one and appending text
            if prev_segment.text != current_segment.text:
                merged_segments[
                    -1
                ].text = f"{prev_segment.text}\n{current_segment.text}"

            # Add current segment's image to the list of unused files
            unused_image_files.append(current_segment.image_path)

            logger.debug(
                f"Merged frame {current_segment.timestamp} "
                f"with {prev_segment.timestamp} (similarity: {similarity:.2f})"
            )
        else:
            # Frames are different, add as a new segment
            merged_segments.append(current_segment)

    return merged_segments, unused_image_files


def delete_unused_image_files(file_paths: list[str]) -> None:
    """Delete unused image files after merging similar frames.

    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted unused image file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")


def compare_images(img1_path: str, img2_path: str) -> float:
    """Compare two images and return a similarity score.

    Args:
        img1_path: Path to the first image
        img2_path: Path to the second image

    Returns:
        Similarity score between 0.0 and 1.0 (higher means more similar)
    """
    try:
        # Use PIL to read images to support all formats (jpg, png, avif, webp)
        pil_img1 = Image.open(img1_path)
        pil_img2 = Image.open(img2_path)

        # Convert PIL images to numpy arrays for OpenCV processing
        img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

        # Resize to the same dimensions if they differ
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale for better comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate histogram-based comparison score
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        # Compare histograms using correlation method
        # Returns a value between 0 and 1 (1 means identical)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    except Exception as e:
        logger.warning(
            f"Failed to read images for comparison: {img1_path}, "
            f"{img2_path}. Error: {e}"
        )
        return 0.0


def _prepare_subtitle_path(subtitle_path: str, output_dir: str) -> str:
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

            for filename in sorted_files:
                file_path = os.path.join(subtitle_path, filename)
                with open(file_path, encoding="utf-8") as infile:
                    content = infile.read()
                    # Remove header if present
                    if content.startswith("WEBVTT"):
                        content = re.sub(
                            r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL
                        )
                    outfile.write(content)
                    outfile.write("\n\n")
        return combined_subtitle_path

    return subtitle_path


def _save_frame_image(
    frame: np.ndarray,
    image_path: str,
    image_format: Literal["jpg", "png", "avif", "webp"],
) -> None:
    """Save a video frame in the specified image format.

    Args:
        frame: The video frame as a numpy array (OpenCV format)
        image_path: Path where the image should be saved
        image_format: Format to save the image (jpg, png, avif, webp)
    """
    if image_format in ("avif", "webp"):
        # Convert OpenCV BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(image_path, format=image_format.upper(), quality=90)
        logger.debug(f"Saved frame as {image_format.upper()}: {image_path}")
    else:
        # Use OpenCV for jpg and png
        cv2.imwrite(image_path, frame)
        logger.debug(f"Saved frame as {image_format.upper()}: {image_path}")
