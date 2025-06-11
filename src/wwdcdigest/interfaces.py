"""Interfaces for WWDC Digest components based on SOLID principles."""

import abc
from typing import Literal, Protocol, runtime_checkable

from .models import OpenAIConfig, WWDCDigest, WWDCFrameSegment


@runtime_checkable
class VideoProcessor(Protocol):
    """Interface for video processing components."""

    async def extract_frames(
        self,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        image_format: Literal["jpg", "png", "avif", "webp"] = "jpg",
    ) -> list[WWDCFrameSegment]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_dir: Directory to save extracted frames
            image_format: Format for the extracted images

        Returns:
            List of WWDCFrameSegment objects
        """
        ...


@runtime_checkable
class ContentSummarizer(Protocol):
    """Interface for content summarization components."""

    async def generate_summary(
        self,
        transcript: str,
        session_title: str,
        language: str = "en",
    ) -> tuple[str, list[str]]:
        """Generate a summary and key points from a transcript.

        Args:
            transcript: The transcript text
            session_title: Title of the session
            language: Language code for the output

        Returns:
            Tuple of (summary, key_points)
        """
        ...


@runtime_checkable
class ContentTranslator(Protocol):
    """Interface for content translation components."""

    async def translate(
        self,
        summary: str,
        key_points: list[str],
        segments: list[WWDCFrameSegment],
        target_language: str,
    ) -> tuple[str, list[str], list[WWDCFrameSegment]]:
        """Translate digest content to the target language.

        Args:
            summary: The summary text to translate
            key_points: List of key points to translate
            segments: List of WWDCFrameSegment objects with text to translate
            target_language: Target language code

        Returns:
            Tuple of (translated_summary, translated_key_points, translated_segments)
        """
        ...


@runtime_checkable
class DigestFormatter(Protocol):
    """Interface for digest formatting components."""

    def format_digest(self, digest: WWDCDigest, output_path: str) -> str:
        """Format a digest into a specific output format.

        Args:
            digest: The WWDCDigest object to format
            output_path: Path to save the formatted output

        Returns:
            Path to the created output file
        """
        ...


class OpenAISummarizer(abc.ABC):
    """Abstract base class for OpenAI-based summarizers."""

    def __init__(self, config: OpenAIConfig) -> None:
        """Initialize the summarizer with OpenAI config.

        Args:
            config: OpenAI API configuration
        """
        self.config = config

    @abc.abstractmethod
    async def generate_summary(
        self,
        transcript: str,
        session_title: str,
        language: str = "en",
    ) -> tuple[str, list[str]]:
        """Generate a summary and key points from a transcript.

        Args:
            transcript: The transcript text
            session_title: Title of the session
            language: Language code for the output

        Returns:
            Tuple of (summary, key_points)
        """
        pass


class OpenAITranslator(abc.ABC):
    """Abstract base class for OpenAI-based translators."""

    def __init__(self, config: OpenAIConfig) -> None:
        """Initialize the translator with OpenAI config.

        Args:
            config: OpenAI API configuration
        """
        self.config = config

    @abc.abstractmethod
    async def translate(
        self,
        summary: str,
        key_points: list[str],
        segments: list[WWDCFrameSegment],
        target_language: str,
    ) -> tuple[str, list[str], list[WWDCFrameSegment]]:
        """Translate digest content to the target language.

        Args:
            summary: The summary text to translate
            key_points: List of key points to translate
            segments: List of WWDCFrameSegment objects with text to translate
            target_language: Target language code

        Returns:
            Tuple of (translated_summary, translated_key_points, translated_segments)
        """
        pass
