"""Translator"""

import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
from paddleocr import PaddleOCR, PPStructure
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .utils import fw_fill


FRAME_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
                '#FECB52']
MIN_TEXT_HEIGHT = 100


def _get_frame_color(i):
    """Get the color of the frame."""
    return ImageColor.getrgb(FRAME_COLORS[i % len(FRAME_COLORS)])


class Translator:
    """Translator class.

    Attributes
    ----------
    temp_dir: tempfile.TemporaryDirectory
        Temporary directory for storing translated PDF files
    temp_dir_name: Path
        Path to the temporary directory
    font: ImageFont
        Font for drawing text on the image
    layout_model: PPStructure
        Layout model for detecting text blocks
    ocr_model: PaddleOCR
        OCR model for detecting text in the text blocks
    translate_model: MarianMTModel
        Translation model for translating text
    translate_tokenizer: MarianTokenizer
        Tokenizer for the translation model
    """
    DPI = 300
    FONT_SIZE = 32

    def __init__(self):
        self._load_models()

    def translate_pdf(self, pdf_path: Path, output_path: Path) -> None:
        """Backend function for translating PDF files.

        Translation is performed in the following steps:
            1. Convert the PDF file to images
            2. Detect text blocks in the images
            3. For each text block, detect text and translate it
            4. Draw the translated text on the image
            5. Save the image as a PDF file
            6. Merge all PDF files into one PDF file

        At 3, this function does not translate the text after
        the references section. Instead, saves the image as it is.

        Parameters
        ----------
        pdf_path: Path
            Path to the input PDF file
        output_path: Path
            Path to the output file
        """
        pdf_images = convert_from_path(pdf_path, dpi=self.DPI)

        pdf_files = []
        reached_references = False
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            for i, image in tqdm(enumerate(pdf_images)):
                file_path = temp_dir / f"{i:03}.pdf"
                if not reached_references:
                    ja_image, reached_references = self._translate_one_page(
                        image=image,
                        reached_references=reached_references,
                    )
                    fig, ax = plt.subplots(1, 2, figsize=(20, 14))
                    ax[0].imshow(np.array(image, dtype=np.uint8))
                    ax[1].imshow(np.array(ja_image, dtype=np.uint8))
                    ax[0].axis("off")
                    ax[1].axis("off")
                    plt.tight_layout()
                    plt.savefig(file_path, format="pdf", dpi=self.DPI)
                    plt.close(fig)
                else:
                    (
                        image.convert("RGB")
                        .resize((int(1400 / image.size[1] * image.size[0]), 1400))
                        .save(file_path, format="pdf")
                    )

                pdf_files.append(str(file_path))

            self._merge_pdfs(pdf_files, output_path)

    def _load_models(self):
        """Backend function for loading models.

        Called in the constructor.
        Load the layout model, OCR model, translation model and font.
        """
        self.font = ImageFont.truetype(
            "/home/pdf-translator/SourceHanSerif-Light.otf",
            size=self.FONT_SIZE,
        )

        self.layout_model = PPStructure(table=False, ocr=False, lang="en")
        self.ocr_model = PaddleOCR(ocr=True, lang="en", ocr_version="PP-OCRv3")

        self.translate_model = MarianMTModel.from_pretrained("staka/fugumt-en-ja").to(
            "cuda"
        )
        self.translate_tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    def _translate_one_page(
            self,
            image: Image.Image,
            reached_references: bool,
    ) -> Tuple[np.ndarray, bool]:
        """Translate one page of the PDF file.

        There are some heuristics to clean-up the results of translation:
            1. Remove newlines, tabs, brackets, slashes, and pipes
            2. Reject the result if there are few Japanese characters
            3. Skip the translation if the text block has only one line

        Parameters
        ----------
        image: Image.Image
            Image of the page
        reached_references: bool
            Whether the references section has been reached.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, bool]
            Translated image, original image,
            and whether the references section has been reached.
        """
        ja_image = image.copy()
        result = self.layout_model(np.array(image, dtype=np.uint8))
        en_draw = ImageDraw.Draw(image)
        ja_draw = ImageDraw.Draw(ja_image)
        i = 0
        for line in result:
            if not line["type"] == "title":
                ocr_results = list(map(lambda x: x[0], self.ocr_model(line["img"])[1]))
                x0, y0, x1, y1 = line["bbox"]

                if len(ocr_results) > 1:
                    text = " ".join(ocr_results)
                    text = re.sub(r"\n|\t|\[|\]|\/|\|", " ", text)
                    #print(text)
                    translated_text = self._translate(text)
                    #print(translated_text)

                    # if almost all characters in translated text are not japanese characters, skip
                    if len(
                            re.findall(
                                r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]",
                                translated_text,
                            )
                    ) > 0.8 * len(translated_text):
                        print(f"skipped (non japanese): {translated_text}")
                        continue

                    # if text is too short, skip
                    if len(translated_text) < 20:
                        print(f"skipped (too short) {translated_text}")
                        continue

                    processed_text = fw_fill(
                        translated_text,
                        width=int(
                            (x1 - x0) / (self.FONT_SIZE / 2)
                        )
                              - 1,
                    )
                    #print(processed_text)

                    text_size = ja_draw.multiline_textsize(text=processed_text, font=self.font)
                    original_height = y1 - y0
                    if original_height > MIN_TEXT_HEIGHT and text_size[1] < original_height / 2:
                        # if the translated text block is too small, it may be a figure
                        print(f"skipped (figure?): {processed_text}")
                        continue

                    # draw translated text on the image
                    ja_draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=_get_frame_color(i))
                    ja_draw.multiline_text((x0, y0), text=processed_text, font=self.font, fill=(0, 0, 0))

                    # draw a frame on the original image
                    en_draw.rectangle((x0, y0, x1, y1), outline=_get_frame_color(i))

                    i += 1
            else:
                try:
                    title = self.ocr_model(line["img"])[1][0][0]
                except IndexError:
                    continue
                if title.lower() == "references" or title.lower() == "reference":
                    reached_references = True

        return ja_image, reached_references

    def _translate(self, text: str) -> str:
        """Translate text using the translation model.

        If the text is too long, it will be splited with
        the heuristic that each sentence should be within 448 characters.

        Parameters
        ----------
        text: str
            Text to be translated.

        Returns
        -------
        str
            Translated text.
        """
        texts = self._split_text(text, 448)

        translated_texts = []
        for i, t in enumerate(texts):
            inputs = self.translate_tokenizer(t, return_tensors="pt").input_ids.to(
                "cuda"
            )
            outputs = self.translate_model.generate(inputs, max_length=512)
            res = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # skip weird translations
            if res.startswith("「この版"):
                continue

            translated_texts.append(res)
        print(translated_texts)
        return "".join(translated_texts)

    def _split_text(self, text: str, text_limit: int = 448) -> List[str]:
        """Split text into chunks of sentences within text_limit.

        Parameters
        ----------
        text: str
            Text to be split.
        text_limit: int
            Maximum length of each chunk. Defaults to 448.

        Returns
        -------
        List[str]
            List of text chunks,
            each of which is shorter than text_limit.
        """
        if len(text) < text_limit:
            return [text]

        sentences = text.rstrip().split(". ")
        sentences = [s + ". " for s in sentences[:-1]] + sentences[-1:]
        result = []
        current_text = ""
        for sentence in sentences:
            if len(current_text) + len(sentence) < text_limit:
                current_text += sentence
            else:
                if current_text:
                    result.append(current_text)
                while len(sentence) >= text_limit:
                    # better to look for a white space at least?
                    result.append(sentence[:text_limit - 1])
                    sentence = sentence[text_limit - 1:].lstrip()
                current_text = sentence
        if current_text:
            result.append(current_text)
        return result

    def _merge_pdfs(self, pdf_files: List[str], output_path: Path) -> None:
        """Merge translated PDF files into one file.

        Merged file will be stored in the temp directory
        as "translated.pdf".

        Parameters
        ----------
        pdf_files: List[str]
            List of paths to translated PDF files stored in
            the temp directory.
        """
        pdf_merger = PyPDF2.PdfMerger()
        for pdf_file in sorted(pdf_files):
            pdf_merger.append(pdf_file)
        pdf_merger.write(output_path)
