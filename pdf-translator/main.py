import logging
from pathlib import Path

import click

from .translator import Translator


def translate(filename: Path) -> None:
    """Translate the given PDF file."""
    translator = Translator()
    output_path = filename.parent / (filename.stem + '_ja.pdf')
    translator.translate_pdf(filename, output_path)


@click.command()
@click.argument('filename', required=True, type=click.Path(exists=True))
@click.option('--log-level', default='INFO')
def main(filename, log_level) -> None:
    """Translates a PDF file."""
    log_level = getattr(logging, log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=log_level)
    translate(Path(filename))


if __name__ == "__main__":
    main()
