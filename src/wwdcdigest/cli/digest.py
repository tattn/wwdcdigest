"""Digest command for the CLI."""

import asyncio
import logging
import os
import sys
from typing import Literal, cast

import click

from wwdcdigest.digest import create_digest
from wwdcdigest.models import AIConfig, DigestOptions, ImageOptions, OpenAIConfig

AIProvider = Literal["none", "openai", "codex", "claude", "command"]

logger = logging.getLogger("wwdcdigest")


@click.command("digest")
@click.argument("url", required=True)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for files (creates a session directory inside this)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown"]),
    default="markdown",
    help="Output format (currently only markdown is supported)",
)
@click.option(
    "--ai",
    "ai_provider",
    type=click.Choice(["none", "openai", "codex", "claude", "command"]),
    default=None,
    help=(
        "AI backend for summary/translation. "
        "Use codex or claude to call the installed CLI."
    ),
)
@click.option(
    "--ai-model",
    type=str,
    help="Model name passed to the selected AI backend (optional)",
)
@click.option(
    "--ai-command",
    type=str,
    help=(
        "Custom AI command for --ai command. Use {prompt} as a placeholder, "
        "or the prompt is appended as the final argument."
    ),
)
@click.option(
    "--ai-timeout",
    type=int,
    default=300,
    show_default=True,
    help="Timeout in seconds for external AI CLI calls",
)
@click.option(
    "--openai-key",
    type=str,
    help="OpenAI API key to generate summary and key points (optional)",
)
@click.option(
    "--openai-endpoint",
    type=str,
    help=(
        "Custom OpenAI API endpoint URL (optional). "
        "Can also be set with OPENAI_API_ENDPOINT environment variable."
    ),
)
@click.option(
    "--language",
    "-l",
    type=str,
    default="en",
    help=(
        "Language code for the digest (e.g., 'en', 'ja', 'zh', 'fr'). "
        "Non-English requires an AI backend."
    ),
)
@click.option(
    "--image-format",
    "-i",
    type=click.Choice(["jpg", "png", "avif", "webp"]),
    default="jpg",
    help="Format for extracted images (jpg, png, avif, webp)",
)
@click.option(
    "--image-width",
    "-w",
    type=int,
    default=None,
    help="Width for extracted images in pixels (maintains aspect ratio)",
)
@click.option(
    "--force",
    "-F",
    is_flag=True,
    default=False,
    help="Force regeneration even if digest already exists",
)
def digest_command(  # noqa: PLR0913
    url: str,
    output_dir: str | None,
    output_format: str,
    ai_provider: AIProvider | None,
    ai_model: str | None,
    ai_command: str | None,
    ai_timeout: int,
    openai_key: str | None,
    openai_endpoint: str | None,
    language: str,
    image_format: str,
    image_width: int | None,
    force: bool,
) -> None:
    """Create a digest from a WWDC session.

    URL should be a WWDC session URL
    (e.g., 'https://developer.apple.com/videos/play/wwdc2023/110173/'
    or 'https://developer.apple.com/jp/videos/play/wwdc2023/110173/')
    """
    try:
        if output_format != "markdown":
            raise ValueError("Currently, only 'markdown' format is supported.")

        # Create AIConfig/OpenAIConfig objects if AI options are provided.
        ai_config = None
        openai_config = None
        if openai_key:
            openai_config = OpenAIConfig(api_key=openai_key, endpoint=openai_endpoint)
        if ai_provider:
            ai_config = AIConfig(
                provider=ai_provider,
                model=ai_model,
                api_key=openai_key,
                endpoint=openai_endpoint,
                command=ai_command,
                timeout_seconds=ai_timeout,
            )
        elif openai_key:
            ai_config = AIConfig(
                provider="openai",
                model=ai_model,
                api_key=openai_key,
                endpoint=openai_endpoint,
                timeout_seconds=ai_timeout,
            )

        # Create ImageOptions object
        image_options = ImageOptions(
            format=cast(Literal["jpg", "png", "avif", "webp"], image_format),
            width=image_width,
        )

        # Create DigestOptions object
        digest_options = DigestOptions(
            output_dir=output_dir,
            openai_config=openai_config,
            ai_config=ai_config,
            language=language,
            image_options=image_options,
            force_regenerate=force,
        )

        digest = asyncio.run(
            create_digest(
                url=url,
                options=digest_options,
            )
        )
        logger.info(f"Successfully created digest for session {digest.session.id}")

        # Print output path
        click.echo(f"Digest created: {digest.markdown_path}")

        if os.path.exists(digest.markdown_path):
            click.echo("Open the file to view the digest with embedded frames.")

    except Exception as e:
        logger.error(f"Error creating digest: {e}")
        sys.exit(1)
