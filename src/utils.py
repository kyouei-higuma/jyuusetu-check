"""
画像処理などのユーティリティ
"""
from PIL import Image


def crop_evidence_region(
    pil_image: Image.Image,
    box_2d: list[float | int],
    padding_ratio_y: float = 0.5,
    padding_ratio_x: float = 0.3,
) -> Image.Image:
    """
    正規化座標（0-1000）で指定された矩形で画像を切り抜く。
    上下方向に padding_ratio_y、左右方向に padding_ratio_x の余白を付けて範囲を拡張する。

    Args:
        pil_image: 元のPIL画像
        box_2d: [ymin, xmin, ymax, xmax] の形式。画像を0-1000に正規化した座標系。
        padding_ratio_y: 矩形の縦幅に対する上下の余白の割合（0.5 = 50%）。
        padding_ratio_x: 矩形の横幅に対する左右の余白の割合（0.3 = 30%）。
        画像外にはみ出さないようクランプする。

    Returns:
        切り抜いたPIL画像。座標が無効な場合や範囲外の場合は可能な範囲でクロップした画像。
    """
    if not box_2d or len(box_2d) != 4:
        return pil_image

    ymin_n, xmin_n, ymax_n, xmax_n = box_2d
    w, h = pil_image.size

    # 上下に縦幅の50%、左右に横幅の30%の余白を追加（正規化座標で拡張）
    box_h = ymax_n - ymin_n
    box_w = xmax_n - xmin_n
    pad_y = box_h * padding_ratio_y
    pad_x = box_w * padding_ratio_x

    ymin_n = ymin_n - pad_y
    xmin_n = xmin_n - pad_x
    ymax_n = ymax_n + pad_y
    xmax_n = xmax_n + pad_x

    # 正規化座標を 0-1000 内にクランプ
    ymin_n = max(0.0, min(ymin_n, 1000.0))
    xmin_n = max(0.0, min(xmin_n, 1000.0))
    ymax_n = max(ymin_n + 1.0, min(ymax_n, 1000.0))
    xmax_n = max(xmin_n + 1.0, min(xmax_n, 1000.0))

    # 0-1000 をピクセル座標に変換
    ymin_px = int((ymin_n / 1000.0) * h)
    xmin_px = int((xmin_n / 1000.0) * w)
    ymax_px = int((ymax_n / 1000.0) * h)
    xmax_px = int((xmax_n / 1000.0) * w)

    # 画像範囲内にクランプ（最低1px幅・高さを確保）
    xmin_px = max(0, min(xmin_px, w - 1))
    ymin_px = max(0, min(ymin_px, h - 1))
    xmax_px = max(xmin_px + 1, min(xmax_px, w))
    ymax_px = max(ymin_px + 1, min(ymax_px, h))

    return pil_image.crop((xmin_px, ymin_px, xmax_px, ymax_px))
