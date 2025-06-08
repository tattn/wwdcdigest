"""Digest command for the CLI."""

import asyncio
import logging
import os
import sys

import click

from wwdcdigest.digest import create_digest

from .utils import validate_session_id_or_url

logger = logging.getLogger("wwdcdigest")


@click.command("digest")
@click.argument("url", callback=validate_session_id_or_url, required=True)
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
        "Non-English requires OpenAI."
    ),
)
def digest_command(  # noqa: PLR0913
    url: str,
    output_dir: str | None,
    output_format: str,
    openai_key: str | None,
    openai_endpoint: str | None,
    language: str,
) -> None:
    """Create a digest from a WWDC session.

    URL must be a full URL from developer.apple.com
    (e.g., 'https://developer.apple.com/videos/play/wwdc2023/110173/')
    """
    try:
        if output_format != "markdown":
            raise ValueError("Currently, only 'markdown' format is supported.")

        digest = asyncio.run(
            create_digest(
                url=url,
                output_dir=output_dir,
                openai_key=openai_key,
                openai_endpoint=openai_endpoint,
                language=language,
            )
        )
        logger.info(f"Successfully created digest for session {digest.session_id}")

        # Print output path
        click.echo(f"Digest created: {digest.markdown_path}")

        if os.path.exists(digest.markdown_path):
            click.echo("Open the file to view the digest with embedded frames.")

    except Exception as e:
        logger.error(f"Error creating digest: {e}")
        sys.exit(1)
