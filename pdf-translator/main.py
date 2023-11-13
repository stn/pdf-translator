import argparse
from pathlib import Path

from .translator import Translator


def translate(input_pdf_path: Path, output_dir: Path, prefix="ja_") -> None:
    """Sends a POST request to the translator server to translate a PDF.

    Parameters
    ----------
    input_pdf_path : Path
        Path to the PDF to be translated.
    output_dir : Path
        Path to the directory where the translated PDF will be saved.
    prefix : str
        Prefix to be added to the translated PDF filename. (default: "ja_")
    """
    translator = Translator()
    translator.translate_pdf(input_pdf_path, output_dir / (prefix + input_pdf_path.name))


def main(args: argparse.Namespace) -> None:
    """Translates a PDF or all PDFs in a directory.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the script.

    Raises
    ------
     ValueError
        If the input path is not a valid path to file or directory.

    Notes
    -----
    args must have the following attributes:
        input_pdf_path_or_dir : Path
            Path to the PDF or directory of PDFs to be translated.
        output_dir : Path
            Path to the directory where the translated PDFs
            will be saved.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_pdf_path_or_dir.is_file():
        if args.input_pdf_path_or_dir.suffix != ".pdf":
            raise ValueError(
                f"Input file must be a PDF or directory: {args.input_pdf_path_or_dir}"
            )

        translate(args.input_pdf_path_or_dir, args.output_dir)

    elif args.input_pdf_path_or_dir.is_dir():
        input_pdf_paths = args.input_pdf_path_or_dir.glob("*.pdf")

        if not input_pdf_paths:
            raise ValueError(f"Input directory is empty: {args.input_pdf_path_or_dir}")

        for input_pdf_path in input_pdf_paths:
            translate(input_pdf_path, args.output_dir)

    else:
        raise ValueError(
            f"Input path must be a file or directory: {args.input_pdf_path_or_dir}"
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_pdf_path_or_dir",
        type=Path,
        required=True,
        help="Path to the PDF or directory of PDFs to be translated.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default="./outputs",
        help="Path to the directory where the translated PDFs will be saved. (default: ./outputs)",
    )
    args = parser.parse_args()
    main(args)
