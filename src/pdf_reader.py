"""
PDFを画像化するモジュール
PDFの全ページをJPEG画像に変換し、Base64エンコードされた文字列のリストとして返す。
"""
import base64
import io
from typing import BinaryIO

from PIL import Image
import fitz


def _pixmap_to_jpeg_b64(pix: "fitz.Pixmap") -> str:
    """PixmapをJPEGバイト列に変換しBase64で返す。"""
    buf = io.BytesIO()
    if hasattr(pix, "pil_save"):
        pix.pil_save(buf, format="jpeg", quality=85)
    else:
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pdf_to_images(file_stream: bytes | BinaryIO) -> list[str]:
    """
    PDFの全ページを読み込み、画像データ(JPEG)に変換し、
    Base64エンコードされた文字列のリストとして返す。

    Args:
        file_stream: PDFのバイト列、または読み取り可能なバイナリストリーム

    Returns:
        各ページのJPEG画像をBase64エンコードした文字列のリスト
    """
    if isinstance(file_stream, bytes):
        data = file_stream
    else:
        data = file_stream.read()

    images_b64: list[str] = []
    doc = fitz.open(stream=data, filetype="pdf")

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # 200 DPI: フォームの細かい文字（宅地建物取引士名・登録番号等）を確実に読み取るため
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images_b64.append(_pixmap_to_jpeg_b64(pix))
    finally:
        doc.close()

    return images_b64
