import logging
from pathlib import Path

import click

from .translator import Translator


@click.command()
@click.argument('filename', required=True, type=click.Path(exists=True))
@click.option('--log-level', default='INFO')
@click.option('--font-size', default=8)
@click.option('--dpi', default=300)
def translate(filename, log_level, font_size, dpi) -> None:
    """Translates a PDF file."""
    log_level = getattr(logging, log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=log_level)
    input_path = Path(filename)
    output_path = input_path.parent / (input_path.stem + '_ja.pdf')
    translator = Translator(dpi=dpi, font_size=font_size)
    translator.translate_pdf(input_path, output_path)


if __name__ == "__main__":
    translate()
