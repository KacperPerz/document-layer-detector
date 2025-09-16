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


def draw_comparison(
    image_bgr,
    preds: Iterable,
    gts: Iterable,
    eval_result: dict,
):
    """
    Draws ground truth and predictions on the same image.
    - Matched predictions: green boxes
    - Unmatched predictions (FP): red boxes
    - Unmatched ground truths (FN): yellow boxes
    - All ground truths also outlined in blue for reference
    """
    annotated = image_bgr.copy()

    # Always draw GT in blue
    blue = (255, 0, 0)
    for gt in gts:
        x1, y1, x2, y2 = [int(v) for v in gt["bbox"]]
        label = str(gt.get("label", "GT"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), blue, 2)
        _draw_label(annotated, (x1, y1), f"GT {label}", blue)

    # Draw matched predictions in green
    green = (0, 255, 0)
    for p_idx, g_idx, iou in eval_result.get("matches", []):
        p = preds[p_idx]
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        label = str(p.get("label", "Pred"))
        score = p.get("score")
        caption = f"{label}" + (f" {score:.2f}" if isinstance(score, (float, int)) else "") + f" IoU {iou:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), green, 2)
        _draw_label(annotated, (x1, y1), caption, green)

    # Draw unmatched predictions (FP) in red
    red = (0, 0, 255)
    for p_idx in eval_result.get("unmatched_predictions", []):
        p = preds[p_idx]
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        label = str(p.get("label", "Pred"))
        score = p.get("score")
        caption = f"FP {label}" + (f" {score:.2f}" if isinstance(score, (float, int)) else "")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 2)
        _draw_label(annotated, (x1, y1), caption, red)

    # Draw unmatched ground truths (FN) highlighted in yellow overlay
    yellow = (0, 255, 255)
    for g_idx in eval_result.get("unmatched_ground_truth", []):
        g = gts[g_idx]
        x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
        label = str(g.get("label", "GT"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), yellow, 2)
        _draw_label(annotated, (x1, y1), f"FN {label}", yellow)

    # Footer with metrics summary
    footer = _compose_footer(eval_result)
    if footer:
        annotated = _draw_footer(annotated, footer)

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


def _compose_footer(eval_result: dict):
    try:
        p = eval_result.get("precision", 0.0)
        r = eval_result.get("recall", 0.0)
        f1 = eval_result.get("f1", 0.0)
        miou = eval_result.get("mean_iou", 0.0)
        ap = eval_result.get("ap50", 0.0)
        tp = eval_result.get("tp", 0)
        fp = eval_result.get("fp", 0)
        fn = eval_result.get("fn", 0)
        return f"P {p:.2f} | R {r:.2f} | F1 {f1:.2f} | mIoU {miou:.2f} | AP50 {ap:.2f} | TP {tp} FP {fp} FN {fn}"
    except Exception:
        return None


def _draw_footer(img, text: str):
    if not text:
        return img
    h, w = img.shape[:2]
    footer_h = 26
    bg = img.copy()
    import numpy as _np
    footer = _np.zeros((footer_h, w, 3), dtype=img.dtype)
    footer[:] = (32, 32, 32)
    combined = _np.vstack([img, footer])
    cv2.putText(combined, text, (6, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return combined
