"""Video processing utilities for WWDC digests."""

import logging
import os

import cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401
import webvtt
from PIL import Image

from .models import ImageOptions, WWDCFrameSegment
from .webvtt_utils import (
    parse_webvtt_time,
    prepare_combined_subtitle,
    prepare_subtitle_path,
)

logger = logging.getLogger("wwdcdigest")

# Constants for image comparison
# Higher value means images need to be more similar to be merged
SIMILARITY_THRESHOLD = 0.95


def extract_frames_from_video(
    video_path: str,
    subtitle_path: str,
    output_dir: str,
    image_options: ImageOptions,
) -> list[WWDCFrameSegment]:
    """Extract frames from video at subtitle timestamps.

    Args:
        video_path: Path to the video file
        subtitle_path: Path to the WebVTT subtitle file
        output_dir: Directory to save extracted frames
        image_options: Options for image extraction and formatting

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
    subtitle_path = prepare_subtitle_path(subtitle_path, output_dir)

    # Process subtitle file to remove duplicates
    combined_subtitle_path = prepare_combined_subtitle(subtitle_path, output_dir)

    # Parse WebVTT file
    try:
        # Parse WebVTT
        vtt = webvtt.read(combined_subtitle_path)
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
                image_filename = f"frame_{i:04d}.{image_options.format}"
                image_path = os.path.join(output_dir, image_filename)
                _save_frame_image(frame, image_path, image_options)

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
            # Frames are similar, merge by keeping the last one and appending text
            if prev_segment.text != current_segment.text:
                merged_segments[
                    -1
                ].text = f"{prev_segment.text}\n{current_segment.text}"

            # Keep the last image and mark the previous one as unused
            unused_image_files.append(prev_segment.image_path)
            merged_segments[-1].image_path = current_segment.image_path

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


def _save_frame_image(
    frame: np.ndarray,
    image_path: str,
    image_options: ImageOptions,
) -> None:
    """Save a video frame in the specified image format.

    Args:
        frame: The video frame as a numpy array (OpenCV format)
        image_path: Path where the image should be saved
        image_options: Options for image extraction and formatting
    """
    # Resize if image_width is specified
    if image_options.width is not None:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        new_height = int(image_options.width / aspect_ratio)
        frame = cv2.resize(frame, (image_options.width, new_height))
        logger.debug(f"Resized frame to {image_options.width}x{new_height}")

    if image_options.format in ("avif", "webp"):
        # Convert OpenCV BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(image_path, format=image_options.format.upper(), quality=90)
        logger.debug(f"Saved frame as {image_options.format.upper()}: {image_path}")
    else:
        # Use OpenCV for jpg and png
        cv2.imwrite(image_path, frame)
        logger.debug(f"Saved frame as {image_options.format.upper()}: {image_path}")


def load_segments_from_frames_dir(
    frames_dir: str,
) -> list[WWDCFrameSegment]:
    """Load existing frame segments from a directory.

    Args:
        frames_dir: Directory containing frame images

    Returns:
        List of WWDCFrameSegment objects
    """
    logger.info(f"Loading existing frames from {frames_dir}")
    segments = []

    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        logger.warning(f"Frames directory does not exist: {frames_dir}")
        return segments

    # Find all image files in the directory
    image_extensions = (".jpg", ".jpeg", ".png", ".avif", ".webp")
    image_files = [
        f
        for f in sorted(os.listdir(frames_dir))
        if f.lower().startswith("frame_") and f.lower().endswith(image_extensions)
    ]

    if not image_files:
        logger.warning(f"No frame images found in {frames_dir}")
        return segments

    # Load metadata file if it exists
    metadata_path = os.path.join(frames_dir, "metadata.txt")
    metadata = {}

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as file:
                lines = file.readlines()
                current_frame = None
                current_text = []

                for line_text in lines:
                    line_content = line_text.strip()
                    if line_content.startswith("Frame:"):
                        # Save previous frame data if exists
                        if current_frame and current_text:
                            metadata[current_frame] = "\n".join(current_text)
                            current_text = []

                        # Start new frame
                        current_frame = line_content.replace("Frame:", "").strip()
                    elif current_frame:
                        current_text.append(line_content)

                # Save last frame
                if current_frame and current_text:
                    metadata[current_frame] = "\n".join(current_text)
        except Exception as e:
            logger.error(f"Error reading metadata file: {e}")

    # Process each image file
    for img_file in image_files:
        try:
            # Extract frame number from filename (frame_XXXX.ext)
            frame_number = img_file.split("_")[1].split(".")[0]

            # Full path to the image
            image_path = os.path.join(frames_dir, img_file)

            # Get text from metadata if available
            text = metadata.get(
                frame_number, metadata.get(img_file, f"Frame {frame_number}")
            )

            # Create timestamp (use frame number as timestamp if not available)
            frame_num = int(frame_number)
            timestamp = f"{frame_num // 60:02d}:{frame_num % 60:02d}.000"

            # Create segment
            segment = WWDCFrameSegment(
                timestamp=timestamp, text=text, image_path=image_path
            )

            segments.append(segment)

        except Exception as e:
            logger.warning(f"Error processing frame file {img_file}: {e}")

    logger.info(f"Loaded {len(segments)} frames from directory")
    return segments
