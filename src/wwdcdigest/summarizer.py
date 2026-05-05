"""Implementations of content summarization components."""

import logging

from .cli_ai import complete_json_with_cli
from .interfaces import ExternalAISummarizer, OpenAISummarizer
from .models import OpenAIResponse
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


class ExternalAIContentSummarizer(ExternalAISummarizer):
    """Implementation of ContentSummarizer using an external AI CLI."""

    async def generate_summary(
        self,
        transcript: str,
        session_title: str,
        language: str = "en",
    ) -> tuple[str, list[str]]:
        """Generate a summary and key points using an external AI CLI."""
        logger.info(
            "Generating summary for %s using %s",
            session_title,
            self.config.provider,
        )
        prompt = (
            f"Here's a transcript from the WWDC session titled '{session_title}'.\n\n"
            f"{transcript}\n\n"
            f"Analyze this transcript and respond in language code '{language}'. "
            "Return JSON with exactly these fields: "
            "`summary` as a concise 2-3 paragraph string, and `key_points` as "
            "an array of 3-5 important technical points."
        )
        response = await complete_json_with_cli(prompt, self.config, OpenAIResponse)
        if not isinstance(response, OpenAIResponse):
            raise TypeError("External AI returned an unexpected response type")
        return response.summary, response.key_points
