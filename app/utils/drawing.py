import cv2
from typing import Iterable
from ..core.model import get_label_name


def draw_detections(image_bgr, layout: Iterable):
    """
    Draws bounding boxes and labels on a copy of the input BGR image.
    layout: iterable of layoutparser TextBlock-like objects with .block and .type/.score
    Returns a new BGR image with annotations.
    """
    annotated = image_bgr.copy()

    for block in layout:
        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)
        x2 = int(block.block.x_2)
        y2 = int(block.block.y_2)
        label = get_label_name(block.type)
        score = block.score if hasattr(block, 'score') else None
        color = _color_from_label(label)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}" + (f" {score:.2f}" if isinstance(score, (float, int)) else "")
        _draw_label(annotated, (x1, y1), caption, color)

    return annotated


def _draw_label(img, topleft, text, color):
    x, y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    cv2.rectangle(img, (x, max(0, y - th - 2*pad)), (x + tw + 2*pad, y), color, -1)
    cv2.putText(img, text, (x + pad, max(0, y - pad)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _color_from_label(label: str):
    # Generate a deterministic color from label text
    h = abs(hash(label)) % 360
    return _hsv_to_bgr(h, 200, 255)


def _hsv_to_bgr(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s/255.0, v/255.0)
    return (int(b*255), int(g*255), int(r*255))
