"""Implementations of content summarization components."""

import logging

from .interfaces import OpenAISummarizer
from .openai_utils import generate_summary_and_key_points

logger = logging.getLogger("wwdcdigest")


class DefaultSummarizer:
    """Default implementation that returns placeholder summary and key points."""

    async def generate_summary(
        self,
        transcript: str,  # noqa: ARG002
        session_title: str,
        language: str = "en",  # noqa: ARG002
    ) -> tuple[str, list[str]]:
        """Generate a placeholder summary and key points.

        Args:
            transcript: The transcript text (unused)
            session_title: Title of the session
            language: Language code for the output (unused)

        Returns:
            Tuple of (summary, key_points)
        """
        logger.info(f"Using default summarizer for {session_title}")
        summary = f"Summary of {session_title}"
        key_points = []
        return summary, key_points


class OpenAIContentSummarizer(OpenAISummarizer):
    """Implementation of ContentSummarizer using OpenAI API."""

    async def generate_summary(
        self,
        transcript: str,
        session_title: str,
        language: str = "en",
    ) -> tuple[str, list[str]]:
        """Generate a summary and key points using OpenAI.

        Args:
            transcript: The transcript text
            session_title: Title of the session
            language: Language code for the output

        Returns:
            Tuple of (summary, key_points)
        """
        logger.info(f"Generating summary for {session_title} using OpenAI")
        return await generate_summary_and_key_points(
            transcript, session_title, self.config, language
        )
