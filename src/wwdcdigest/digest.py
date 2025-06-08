"""Functions for creating digests from WWDC sessions.

Environment variables:
    OPENAI_API_KEY: API key for OpenAI (required for non-English digests)
    OPENAI_API_ENDPOINT: Custom OpenAI API endpoint URL (optional)
"""

import logging
import os
import tempfile

from wwdctools.downloader import download_session_content
from wwdctools.models import WWDCSession
from wwdctools.session import fetch_session_data

from .markdown_utils import create_markdown
from .models import OpenAIConfig, WWDCDigest, WWDCFrameSegment
from .openai_utils import generate_summary_and_key_points, translate_text
from .video import extract_frames_from_video

logger = logging.getLogger("wwdcdigest")


async def _get_transcript_from_session(
    download_paths: dict[str, str], segments: list[WWDCFrameSegment]
) -> str:
    """Extract transcript text from session download paths or segments.

    Args:
        download_paths: Dictionary of downloaded file paths
        segments: List of WWDCFrameSegment objects

    Returns:
        Transcript text or empty string if not available
    """
    # First try to use transcript if available
    if "transcript" in download_paths:
        transcript_path = download_paths["transcript"]
        try:
            with open(transcript_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading transcript file: {e}")

    # If no transcript, build one from WebVTT segments
    if segments:
        try:
            captions_text = [segment.text for segment in segments]
            return "\n".join(captions_text)
        except Exception as e:
            logger.error(f"Error creating transcript from captions: {e}")

    return ""


def _validate_openai_settings(
    openai_key: str | None,
    openai_endpoint: str | None,
    language: str,
) -> OpenAIConfig | None:
    """Validate and retrieve OpenAI API settings.

    Args:
        openai_key: OpenAI API key (if provided)
        openai_endpoint: OpenAI API endpoint URL (if provided)
        language: Language code for the digest

    Returns:
        OpenAIConfig object if valid settings are provided, None otherwise

    Raises:
        ValueError: If non-English language is requested but no OpenAI key is available
    """
    # Check for OpenAI API key in environment if not provided
    if not openai_key:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            logger.debug("Using OpenAI API key from environment")

    # Check for OpenAI API endpoint in environment if not provided
    if not openai_endpoint:
        openai_endpoint = os.environ.get("OPENAI_API_ENDPOINT")
        if openai_endpoint:
            logger.debug("Using OpenAI API endpoint from environment")

    # Check if translation is needed but no OpenAI key is available
    if language != "en" and not openai_key:
        raise ValueError(
            "OpenAI API key is required for non-English languages. "
            "Please provide --openai-key or set OPENAI_API_KEY environment variable."
        )

    # Return None if no OpenAI key is available
    if not openai_key:
        return None

    # Create and return OpenAIConfig object
    return OpenAIConfig(api_key=openai_key, endpoint=openai_endpoint)


def _setup_output_directory(
    output_dir: str | None, session_id: str, session_year: str
) -> tuple[str, str, str, str]:
    """Set up output directories.

    Args:
        output_dir: Directory to save output files
        session_id: Session ID
        session_year: Year of the session

    Returns:
        Tuple of (output_dir, session_dir, frames_dir, markdown_path)
    """
    # Create temporary directory if output_dir not specified
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="wwdcdigest_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.debug(f"Using output directory: {output_dir}")

    # Create subdirectory for session in the format wwdc_YEAR_ID
    session_dir = os.path.join(output_dir, f"wwdc_{session_year}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    # Create frames directory
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # The markdown path will be set later using the session title
    markdown_path = ""

    return output_dir, session_dir, frames_dir, markdown_path


async def _handle_file_move(
    old_path: str, expected_path: str, is_dir: bool = False
) -> str:
    """Handle moving a file or directory to its expected location.

    Args:
        old_path: Current path of the file/directory
        expected_path: Target path for the file/directory
        is_dir: Whether the path is a directory

    Returns:
        Final path of the file/directory
    """
    if old_path == expected_path or not os.path.exists(old_path):
        return old_path

    if is_dir and os.path.exists(expected_path):
        # If directory exists, move contents
        for file in os.listdir(old_path):
            os.rename(
                os.path.join(old_path, file),
                os.path.join(expected_path, file),
            )
        os.rmdir(old_path)
    else:
        # Simple rename for files or non-existent target directories
        os.rename(old_path, expected_path)

    return expected_path


async def _organize_downloaded_content(
    temp_download_paths: dict[str, str],
    session_dir: str,
) -> dict[str, str]:
    """Organize downloaded content into standardized paths.

    Args:
        temp_download_paths: Dictionary of temporary download paths
        session_dir: Directory for session files

    Returns:
        Dictionary of final download paths
    """
    download_paths: dict[str, str] = {}

    # Define expected paths
    expected_paths = {
        "video": os.path.join(session_dir, "hd.mp4"),
        "webvtt": os.path.join(session_dir, "webvtt"),
        "transcript": os.path.join(session_dir, "transcript.txt"),
        "sample_code": os.path.join(session_dir, "sample_code.txt"),
    }

    # Process each content type
    for content_type, expected_path in expected_paths.items():
        if content_type in temp_download_paths:
            is_dir = content_type == "webvtt"
            final_path = await _handle_file_move(
                temp_download_paths[content_type], expected_path, is_dir
            )
            download_paths[content_type] = final_path

    return download_paths


async def _check_existing_content(
    session_id: str,
    session_dir: str,
) -> dict[str, str]:
    """Check for existing content files.

    Args:
        session_id: Session ID
        session_dir: Directory for session files

    Returns:
        Dictionary of existing file paths
    """
    expected_video_path = os.path.join(session_dir, "hd.mp4")
    expected_webvtt_dir = os.path.join(session_dir, "webvtt")
    expected_transcript_path = os.path.join(session_dir, "transcript.txt")

    # Check if required files exist
    video_exists = os.path.isfile(expected_video_path)
    webvtt_exists = os.path.isdir(expected_webvtt_dir) and any(
        f.endswith(".webvtt") for f in os.listdir(expected_webvtt_dir)
    )

    if not (video_exists and webvtt_exists):
        return {}

    logger.info(
        f"Video and WebVTT files already exist for session {session_id}, "
        "skipping download"
    )

    download_paths = {
        "video": expected_video_path,
        "webvtt": expected_webvtt_dir,
    }

    # Add transcript if it exists
    if os.path.isfile(expected_transcript_path):
        download_paths["transcript"] = expected_transcript_path

    return download_paths


async def _download_and_extract_frames(
    session_data: WWDCSession, session_dir: str, frames_dir: str
) -> tuple[dict[str, str], list[WWDCFrameSegment]]:
    """Download session content and extract frames.

    Args:
        session_data: Session data
        session_dir: Directory for session files
        frames_dir: Directory for extracted frames

    Returns:
        Tuple of (download_paths, segments)

    Raises:
        ValueError: If required files are not available
    """
    session_id = session_data.id

    # Check for existing content
    download_paths = await _check_existing_content(session_id, session_dir)

    if not download_paths:
        # Download content (video, transcript, WebVTT)
        logger.info(f"Downloading content for session {session_id}")
        temp_download_paths = await download_session_content(
            session_data, session_dir, "hd"
        )

        # Fix nested sample_code path if it exists
        if "sample_code" in temp_download_paths:
            nested_path = temp_download_paths["sample_code"]
            if os.path.dirname(nested_path).endswith(
                f"wwdc_{session_data.year}_{session_id}"
            ):
                correct_path = os.path.join(session_dir, "sample_code.txt")
                if nested_path != correct_path and os.path.exists(nested_path):
                    # Copy content instead of moving to avoid issues with
                    # nested directories
                    with open(nested_path, encoding="utf-8") as src_file:
                        content = src_file.read()
                    with open(correct_path, "w", encoding="utf-8") as dst_file:
                        dst_file.write(content)
                    temp_download_paths["sample_code"] = correct_path

        download_paths = await _organize_downloaded_content(
            temp_download_paths,
            session_dir,
        )

    # Check if required files are available
    if "video" not in download_paths or "webvtt" not in download_paths:
        logger.error("Video or WebVTT files not available")
        raise ValueError("Video or WebVTT files not available for this session")

    # Extract frames
    segments = extract_frames_from_video(
        download_paths["video"], download_paths["webvtt"], frames_dir
    )

    return download_paths, segments


async def _generate_summary_and_key_points(
    config: OpenAIConfig | None,
    download_paths: dict[str, str],
    segments: list[WWDCFrameSegment],
    session_title: str,
    language: str = "en",
) -> tuple[str, list[str]]:
    """Generate summary and key points using OpenAI.

    Args:
        config: OpenAI API configuration
        download_paths: Dictionary of downloaded file paths
        segments: List of WWDCFrameSegment objects
        session_title: Session title
        language: Language code for the output (defaults to English)

    Returns:
        Tuple of (summary, key_points)
    """
    # Set default summary and key points
    summary = f"Summary of {session_title}"
    key_points = []

    # Generate summary and key points if OpenAI API key is provided
    if config:
        logger.info("Generating summary and key points with OpenAI")
        transcript_text = await _get_transcript_from_session(download_paths, segments)

        if transcript_text:
            try:
                summary, key_points = await generate_summary_and_key_points(
                    transcript_text,
                    session_title,
                    config,
                    language,
                )
            except Exception as e:
                logger.error(f"Error generating summary with OpenAI: {e}")
        else:
            logger.warning("No transcript available for summary generation")
    else:
        logger.info(
            "No OpenAI API key provided, skipping summary and key points generation"
        )

    return summary, key_points


async def _translate_content_if_needed(
    language: str,
    config: OpenAIConfig | None,
    segments: list[WWDCFrameSegment],
) -> list[WWDCFrameSegment]:
    """Translate segments to the target language if needed.

    Args:
        language: Target language code
        config: OpenAI API configuration
        segments: List of WWDCFrameSegment objects

    Returns:
        List of WWDCFrameSegment objects with translated text if needed
    """
    if language == "en" or not config:
        return segments

    logger.info(f"Translating content to {language}")
    translated_segments = []

    for segment in segments:
        try:
            translated_text = await translate_text(
                segment.text,
                language,
                config,
            )
            translated_segments.append(
                WWDCFrameSegment(
                    timestamp=segment.timestamp,
                    text=translated_text,
                    image_path=segment.image_path,
                )
            )
        except Exception as e:
            logger.error(f"Error translating segment: {e}")
            # Keep original text if translation fails
            translated_segments.append(segment)

    return translated_segments


async def create_digest_from_url(
    url: str,
    output_dir: str | None = None,
    openai_key: str | None = None,
    openai_endpoint: str | None = None,
    language: str = "en",
) -> WWDCDigest:
    """Create a digest from a WWDC session URL.

    Args:
        url: URL of the WWDC session
        output_dir: Directory to save output files (defaults to temp directory)
        openai_key: OpenAI API key to generate summary and key points (optional)
        openai_endpoint: Custom OpenAI API endpoint URL (optional)
        language: Language code for the digest (defaults to English)

    Returns:
        A digest of the session
    """
    logger.info(f"Creating digest from URL: {url}")

    # Validate and get OpenAI settings
    openai_config = _validate_openai_settings(openai_key, openai_endpoint, language)

    # Fetch session data
    session_data = await fetch_session_data(url)
    session_id = session_data.id

    # Set up directories
    _, session_dir, frames_dir, markdown_path = _setup_output_directory(
        output_dir, session_id, str(session_data.year)
    )

    # Download content and extract frames
    download_paths, segments = await _download_and_extract_frames(
        session_data, session_dir, frames_dir
    )

    # Generate summary and key points
    summary, key_points = await _generate_summary_and_key_points(
        openai_config,
        download_paths,
        segments,
        session_data.title,
        language,
    )

    # Translate content if needed
    segments = await _translate_content_if_needed(language, openai_config, segments)

    # Create markdown file path with title as filename
    title_for_filename = (
        session_data.title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    markdown_path = os.path.join(session_dir, f"{title_for_filename}.md")

    # Create the digest object
    digest = WWDCDigest(
        session_id=session_id,
        title=session_data.title,
        summary=summary,
        key_points=key_points,
        segments=segments,
        markdown_path=markdown_path,
        language=language,
    )

    # Create markdown
    create_markdown(digest, markdown_path)

    return digest


async def create_digest(
    url: str,
    output_dir: str | None = None,
    openai_key: str | None = None,
    openai_endpoint: str | None = None,
    language: str = "en",
) -> WWDCDigest:
    """Create a digest from a WWDC session URL.

    Args:
        url: URL of the WWDC session
        output_dir: Directory to save output files (defaults to temp directory)
        openai_key: OpenAI API key to generate summary and key points (optional)
        openai_endpoint: Custom OpenAI API endpoint URL (optional)
        language: Language code for the digest (defaults to English)

    Returns:
        A digest of the session
    """
    logger.info(f"Creating digest for {url}")

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "URL must start with http:// or https:// and be a valid WWDC session URL"
        )

    return await create_digest_from_url(
        url, output_dir, openai_key, openai_endpoint, language
    )
