"""Translator"""

import logging
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from paddleocr import PaddleOCR, PPStructure
from pdf2image import convert_from_path
from PIL import Image
import pypdf
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from transformers import MarianMTModel, MarianTokenizer

from typing import List


FRAME_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF',
                '#FECB52']
MIN_TEXT_HEIGHT = 100
FONT = 'HeiseiMin-W3'

WS_PAT = re.compile(r"\n|\t|\[|\]|\/|\|")


def _get_frame_color(i: int) -> str:
    """Get the color of the frame."""
    return FRAME_COLORS[i % len(FRAME_COLORS)]


class Translator:
    """Translator class."""

    def __init__(
            self,
            dpi: int = 300,
            font_size: int = 8,
    ):
        self._dpi = dpi
        self._font_size = font_size
        self._load_models()
        self._register_font()

    def _load_models(self):
        """Backend function for loading models."""
        self._layout_model = PPStructure(table=False, ocr=False, lang="en", show_log=False)
        self._ocr_model = PaddleOCR(ocr=True, lang="en", ocr_version="PP-OCRv3")

        self.translate_model = MarianMTModel.from_pretrained("staka/fugumt-en-ja").to("cuda")
        self.translate_tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    def _register_font(self):
        """Register a font for drawing text on PDF files."""
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def translate_pdf(self, pdf_path: Path, output_path: Path) -> None:
        """Backend function for translating PDF files."""
        self._logger.info("Converting PDF to images")
        pdf_images = convert_from_path(pdf_path, dpi=self._dpi)

        reader = pypdf.PdfReader(str(pdf_path))
        writer = pypdf.PdfWriter()

        for i, image in enumerate(pdf_images):
            self._logger.info("Translating page %d", i)
            page = reader.pages[i]
            self._translate_one_page(image, page, writer)

        with open(output_path, "wb") as f:
            writer.write(f)

    def _translate_one_page(
            self,
            image: Image.Image,
            page: pypdf.PageObject,
            writer: pypdf.PdfWriter,
    ):
        """Translate one page of the PDF file."""
        box = page.mediabox
        en_page = writer.add_blank_page(width=box.width, height=box.height)
        en_page.merge_page(page)
        ja_page = writer.add_blank_page()
        ja_page.merge_page(page)

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

        def add_text(i, text, bbox, page):
            # Create a canvas to draw the text
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=(page_width, page_height))
            # Draw a white rectangle over the original text
            c.setFillColorRGB(255, 255, 255)
            c.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], stroke=0, fill=1)
            # Draw the translated text
            style = getSampleStyleSheet()["Normal"]
            style.fontName = FONT
            style.fontSize = self._font_size
            style.borderWidth = 1
            style.backColor = "#ffffff"
            style.borderColor = _get_frame_color(i)
            style.spaceBefore = self._font_size
            style.wordWrap = "CJK"
            p = Paragraph(text, style)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            w, h = p.wrap(w, h)
            p.drawOn(c, bbox[0], bbox[1] - h)
            c.showPage()
            c.save()
            # Merge the canvas with the page
            buffer.seek(0)
            text_page = pypdf.PdfReader(buffer).pages[0]
            page.merge_page(text_page)

        def add_rect(i, bbox, page):
            # Create a canvas to draw the text
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=(page_width, page_height))
            # Draw a rectangle around the original text
            c.setStrokeColor(_get_frame_color(i))
            c.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], stroke=1, fill=0)
            c.showPage()
            c.save()
            # Merge the canvas with the page
            buffer.seek(0)
            text_page = pypdf.PdfReader(buffer).pages[0]
            page.merge_page(text_page)

        layout = self._detect_layout(image)
        for i, line in enumerate(layout):
            if line["type"] == "title":
                continue

            text = self._ocr_image(line["img"])
            translated_text = self._translate(text)

            # if text is too short, skip
            if len(translated_text) < 20:
                self._logger.debug("skipped a short text: %s", translated_text)
                continue

            # if almost all characters in translated text are not japanese characters, skip
            if len(
                    re.findall(
                        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]",
                        translated_text,
                    )
            ) > 0.8 * len(translated_text):
                self._logger.debug(f"skipped (non japanese): {translated_text}")
                continue

            bbox = convert_bbox(line["bbox"])
            add_text(i, translated_text, bbox, ja_page)

            # draw a frame on the original image
            add_rect(i, bbox, en_page)

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
        self._logger.debug("Translating text")
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
