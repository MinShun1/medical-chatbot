import re
from PIL import Image
import pytesseract


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR output by removing extra whitespace.
    """

    text = re.sub(r"\s+", " ", text)
    text = text.replace(" :", ":")
    text = text.strip()

    return text


def run_ocr(image_path: str) -> str:
    """
    Perform OCR on an image using Tesseract.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    str
        Cleaned OCR text.
    """

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return clean_ocr_text(text)