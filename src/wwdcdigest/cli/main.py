"""Command-line interface for WWDC Digest."""

import logging

import click

from wwdcdigest import __version__
from wwdcdigest.logger import setup_logger

from .digest import digest_command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output", default=False
)
@click.option(
    "--quiet", "-q", is_flag=True, help="Suppress non-error messages", default=False
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Write logs to file",
    default=None,
)
def main(verbose: bool, quiet: bool, log_file: str | None) -> None:
    """WWDC Digest - Tools for creating digests from Apple WWDC sessions.

    Run a subcommand with --help to see specific options for that command.
    """
    # Set up logging
    log_level = logging.WARNING
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR

    setup_logger(level=log_level, log_file=log_file)


# Register subcommands
main.add_command(digest_command)


if __name__ == "__main__":
    main()
