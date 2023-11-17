"""Translator"""

import logging
from pathlib import Path
import pprint
import re
import tempfile
from typing import List, Tuple, Union

import numpy as np
from paddleocr import PaddleOCR, PPStructure
from pdf2image import convert_from_path
from PIL import Image
import pypdf
import reportlab as rl
from transformers import MarianMTModel, MarianTokenizer

from .utils import fw_fill

FRAME_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
                '#FECB52']
MIN_TEXT_HEIGHT = 100

WS_PAT = re.compile(r"\n|\t|\[|\]|\/|\|")


def _get_frame_color(i: int) -> str:
    """Get the color of the frame."""
    return FRAME_COLORS[i % len(FRAME_COLORS)]


class Translator:
    """Translator class."""

    def __init__(
            self,
            dpi: int = 300,
            font_size: int = 32
    ):
        self._dpi = dpi
        self._font_size = font_size
        self._load_models()

    def _load_models(self):
        """Backend function for loading models."""
        self._layout_model = PPStructure(table=False, ocr=False, lang="en", show_log=False)
        self._ocr_model = PaddleOCR(ocr=True, lang="en", ocr_version="PP-OCRv3")

        self.translate_model = MarianMTModel.from_pretrained("staka/fugumt-en-ja").to("cuda")
        self.translate_tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    @property
    def _logger(self):
        return logging.getLogger(__name__)

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
        self._logger.info("Converting PDF to images")
        pdf_images = convert_from_path(pdf_path, dpi=self._dpi)

        reader = pypdf.PdfReader(str(pdf_path))
        writer = pypdf.PdfWriter()

        # pdf_files = []
        # reached_references = False
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     temp_dir = Path(temp_dir)
        for i, image in enumerate(pdf_images):
            self._logger.info("Translating page %d", i)

            page = reader.pages[i]
            box = page.mediabox
            en_page = writer.add_blank_page(width=box.width, height=box.height)
            en_page.merge_page(page)
            ja_page = writer.add_blank_page()
            ja_page.merge_page(page)

            # file_path = temp_dir / f"{i:03}.pdf"
            # if not reached_references:
            self._translate_one_page(image, writer, en_page, ja_page)
            # fig, ax = plt.subplots(1, 2, figsize=(20, 14))
            # ax[0].imshow(np.array(image, dtype=np.uint8))
            # ax[1].imshow(np.array(ja_image, dtype=np.uint8))
            # ax[0].axis("off")
            # ax[1].axis("off")
            # plt.tight_layout()
            # plt.savefig(file_path, format="pdf", dpi=self._dpi)
            # plt.close(fig)
            # else:
            #     (
            #         image.convert("RGB")
            #         .resize((int(1400 / image.size[1] * image.size[0]), 1400))
            #         .save(file_path, format="pdf")
            #     )

            # pdf_files.append(str(file_path))

            # self._merge_pdfs(pdf_files, output_path)
        with open(output_path, "wb") as f:
            writer.write(f)

    def _translate_one_page(
            self,
            image: Image.Image,
            writer: pypdf.PdfWriter,
            en_page: pypdf.PageObject,
            ja_page: pypdf.PageObject,
    ) -> np.ndarray:
        """Translate one page of the PDF file."""
        # ja_image = image.copy()
        # en_draw = ImageDraw.Draw(image)
        # ja_draw = ImageDraw.Draw(ja_image)

        image_width = image.size[0]
        image_height = image.size[1]
        page_width = en_page.mediabox.width
        page_height = en_page.mediabox.height
        width_ratio = page_width / image_width
        height_ratio = page_height / image_height

        def convert_bbox(bbox):
            x0 = bbox[0] * width_ratio
            y0 = (image_height - bbox[1]) * height_ratio
            x1 = bbox[2] * width_ratio
            y1 = (image_height - bbox[3]) * height_ratio
            return x0, y0, x1, y1

        def add_text(text, bbox, page):
            text_box = pypdf.annotations.FreeText(
                text=text,
                rect=convert_bbox(bbox),
                font_size=f"{self._font_size}pt",
                font_color="000000",
                border_color=_get_frame_color(i),
                background_color="ffffff",
            )
            writer.add_annotation(page, annotation=text_box)

        def add_rect(bbox, page):
            writer.add_annotation(page, annotation=pypdf.annotations.Rectangle(
                rect=convert_bbox(bbox),
            ))

        layout = self._detect_layout(image)
        for i, line in enumerate(layout):
            if line["type"] == "title":
                continue

            text = self._ocr_image(line["img"])
            translated_text = self._translate(text)

            # if almost all characters in translated text are not japanese characters, skip
            if len(
                    re.findall(
                        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]",
                        translated_text,
                    )
            ) > 0.8 * len(translated_text):
                self._logger.debug(f"skipped (non japanese): {translated_text}")
                continue

            # if text is too short, skip
            if len(translated_text) < 20:
                self._logger.debug("skipped a short text: %s", translated_text)
                continue

            # x0, y0, x1, y1 = line["bbox"]

            # processed_text = fw_fill(
            #     translated_text,
            #     width=int((x1 - x0) / (self._font_size / 2)) - 1,
            # )

            # text_size = ja_draw.multiline_textsize(text=processed_text, font=self.font)
            # original_height = y1 - y0
            # if original_height > MIN_TEXT_HEIGHT and text_size[1] < original_height / 2:
            #     # if the translated text block is too small, it may be a figure
            #     print(f"skipped (figure?): {processed_text}")
            #     continue

            # draw translated text on the image
            # ja_draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=_get_frame_color(i))
            # ja_draw.multiline_text((x0, y0), text=processed_text, font=self.font, fill=(0, 0, 0))

            bbox = line["bbox"]
            add_text(translated_text, bbox, ja_page)

            # draw a frame on the original image
            # en_draw.rectangle((x0, y0, x1, y1), outline=_get_frame_color(i))
            add_rect(bbox, en_page)

    def _ocr_image(self, img: np.ndarray) -> str:
        """OCR the image."""
        self._logger.debug("OCR image")
        ocr_results = self._ocr_model.ocr(img, cls=False)
        if len(ocr_results) == 0 or len(ocr_results[0]) == 0:
            return ""
        ocr_texts = list(map(lambda x: x[1][0], ocr_results[0]))
        text = " ".join(ocr_texts)
        text = WS_PAT.sub(" ", text)
        return text

    def _detect_layout(self, image: Image.Image) -> List[dict]:
        """Detect text blocks in the image."""
        self._logger.debug("Detecting layout")
        return self._layout_model(np.array(image, dtype=np.uint8))

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
            inputs = self.translate_tokenizer(t, return_tensors="pt").input_ids.to("cuda")
            outputs = self.translate_model.generate(inputs, max_length=512)
            res = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # skip weird translations
            if res.startswith("「この版"):
                continue

            translated_texts.append(res)
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

    # def _merge_pdfs(self, pdf_files: List[str], output_path: Path) -> None:
    #     """Merge translated PDF files into one file.
    #
    #     Merged file will be stored in the temp directory
    #     as "translated.pdf".
    #
    #     Parameters
    #     ----------
    #     pdf_files: List[str]
    #         List of paths to translated PDF files stored in
    #         the temp directory.
    #     """
    #     pdf_merger = PyPDF2.PdfMerger()
    #     for pdf_file in sorted(pdf_files):
    #         pdf_merger.append(pdf_file)
    #     pdf_merger.write(output_path)
