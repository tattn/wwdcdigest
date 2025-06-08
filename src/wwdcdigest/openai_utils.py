"""Functions for interacting with OpenAI's API."""

import logging

from openai import APIError, AsyncOpenAI, RateLimitError

from .models import OpenAIConfig, OpenAIResponse

logger = logging.getLogger("wwdcdigest")


class OpenAIError(Exception):
    """Base class for OpenAI-related errors."""

    pass


async def translate_text(
    text: str,
    target_language: str,
    config: OpenAIConfig,
) -> str:
    """Translate text to the target language using OpenAI.

    Args:
        text: The text to translate
        target_language: The target language code (e.g., "ja", "fr", "es")
        config: OpenAI API configuration

    Returns:
        Translated text

    Raises:
        OpenAIError: If there's an error calling the OpenAI API
    """
    logger.info(f"Translating text to {target_language}")

    try:
        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.endpoint,
        )

        # Create a prompt for translation
        prompt = (
            f"Translate the following text into {target_language}. "
            f"Maintain the technical accuracy and terminology:\n\n{text}"
        )

        completion = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert translator specializing in technical "
                        "content. Translate the text accurately while preserving "
                        "technical terms and maintaining the original meaning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # Extract translated text
        if translation := completion.choices[0].message.content:
            logger.debug("Successfully translated text")
            return translation

        logger.error("No translation found in OpenAI completion")
        raise OpenAIError("No translation found in OpenAI completion")

    except RateLimitError as e:
        logger.error("Rate limit exceeded when calling OpenAI API")
        raise OpenAIError("OpenAI API rate limit exceeded") from e

    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIError(f"OpenAI API error: {e}") from e

    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise OpenAIError("An unexpected error occurred while translating") from e


async def generate_summary_and_key_points(
    transcript: str,
    session_title: str,
    config: OpenAIConfig,
    language: str = "en",
) -> tuple[str, list[str]]:
    """Generate a summary and key points from a transcript using OpenAI's API.

    Args:
        transcript: The transcript text
        session_title: The title of the session
        config: OpenAI API configuration
        language: The language code for the output (e.g., "ja", "fr", "es")

    Returns:
        Tuple of (summary, key_points)

    Raises:
        OpenAIError: If there's an error calling the OpenAI API
    """
    logger.info(
        f"Generating summary and key points for '{session_title}' in {language}"
    )

    try:
        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.endpoint,
        )

        # Determine language for output
        language_name = {
            "en": "English",
            "ja": "Japanese",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "pt": "Portuguese",
        }.get(language, language)

        # Create a prompt that will help generate structured JSON output
        prompt = (
            f"Here's a transcript from the WWDC session titled '{session_title}'.\n\n"
            f"{transcript}\n\n"
            f"Please analyze this transcript and provide a response in {language_name}:"
            "\n1. A concise summary (2-3 paragraphs)\n"
            "2. 3-5 key technical points"
        )

        # Create system message with language instruction
        system_content = (
            "You are an expert at summarizing technical presentations "
            "and extracting key technical points. Focus on the most "
            "important technical details and new announcements."
            f" You must write your entire response in {language_name}."
        )

        completion = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format=OpenAIResponse,
        )

        # Parse JSON and validate with Pydantic model
        if response := completion.choices[0].message.parsed:
            logger.debug("Successfully generated and parsed summary and key points")
            return response.summary, response.key_points

        logger.error("No parsed response found in OpenAI completion")
        raise OpenAIError("No parsed response found in OpenAI completion")

    except RateLimitError as e:
        logger.error("Rate limit exceeded when calling OpenAI API")
        raise OpenAIError("OpenAI API rate limit exceeded") from e

    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIError(f"OpenAI API error: {e}") from e

    except Exception as e:
        logger.error(f"Error generating summary and key points: {e}")
        raise OpenAIError(
            "An unexpected error occurred while generating content"
        ) from e
