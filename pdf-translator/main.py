import click
from pathlib import Path

from .translator import Translator


def translate(filename: Path) -> None:
    """Translate the given PDF file."""
    translator = Translator()
    output_path = filename.parent / (filename.stem + '_ja.pdf')
    translator.translate_pdf(filename, output_path)


@click.command()
@click.argument('filename', required=True, type=click.Path(exists=True))
def main(filename) -> None:
    """Translates a PDF file."""
    translate(Path(filename))


if __name__ == "__main__":
    main()
