"""``ariadne`` CLI entrypoint.

The CLI is intentionally tiny in v0.0.1: a click group with ``--version`` and
a no-op ``hello`` subcommand that proves the entrypoint is wired up. Real
subcommands (``ui``, ``eval``, ``bench``, ``export``, ``import``, ``drift``)
are added in their respective phases.
"""

from __future__ import annotations

import click

from ariadne_eval._version import __version__


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="ariadne — trajectory-level observability and evaluation for LLM agents.",
)
@click.version_option(version=__version__, prog_name="ariadne")
def cli() -> None:
    """Top-level ``ariadne`` command group."""


@cli.command()
@click.option("--name", default="world", show_default=True, help="Name to greet.")
def hello(name: str) -> None:
    """Print a friendly greeting; placeholder while the real CLI is built out."""
    click.echo(f"Hello, {name}! ariadne-eval v{__version__} is alive.")


if __name__ == "__main__":  # pragma: no cover
    cli()
