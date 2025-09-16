import json
from typing import List, Dict, Tuple, Optional


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(
    preds: List[Dict],
    gts: List[Dict],
    iou_threshold: float = 0.5,
    require_label_match: bool = True,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Returns (matches, unmatched_pred_idxs, unmatched_gt_idxs)
    matches: list of (pred_idx, gt_idx, iou)
    """
    used_gts = set()
    matches: List[Tuple[int, int, float]] = []

    # Precompute IoUs
    for p_idx, p in enumerate(preds):
        best_iou = 0.0
        best_gt_idx: Optional[int] = None
        for g_idx, g in enumerate(gts):
            if g_idx in used_gts:
                continue
            if require_label_match and (str(p.get("label")) != str(g.get("label"))):
                continue
            iou = bbox_iou(p["bbox"], g["bbox"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_gt_idx is not None:
            used_gts.add(best_gt_idx)
            matches.append((p_idx, best_gt_idx, best_iou))

    matched_pred_idxs = {m[0] for m in matches}
    matched_gt_idxs = {m[1] for m in matches}

    unmatched_pred_idxs = [i for i in range(len(preds)) if i not in matched_pred_idxs]
    unmatched_gt_idxs = [i for i in range(len(gts)) if i not in matched_gt_idxs]

    return matches, unmatched_pred_idxs, unmatched_gt_idxs


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_detections(
    preds: List[Dict],
    gts: List[Dict],
    iou_threshold: float = 0.5,
    require_label_match: bool = True,
) -> Dict:
    """
    preds: [{bbox:[x1,y1,x2,y2], label:str, score:float}]
    gts:   [{bbox:[x1,y1,x2,y2], label:str}]
    returns metrics and matching info
    """
    # Sort predictions by score descending for AP computation
    preds_sorted = sorted(preds, key=lambda d: float(d.get("score", 0.0)), reverse=True)

    # One-threshold matching for P/R/F1 and mean IoU
    matches, unmatched_p, unmatched_g = greedy_match(
        preds_sorted, gts, iou_threshold=iou_threshold, require_label_match=require_label_match
    )
    tp = len(matches)
    fp = len(unmatched_p)
    fn = len(unmatched_g)
    prf = precision_recall_f1(tp, fp, fn)
    mean_iou = sum(m[2] for m in matches) / tp if tp > 0 else 0.0

    # Simple AP@IoU via PR curve over score thresholds
    pr_points: List[Tuple[float, float]] = []
    if len(preds_sorted) > 0:
        thresholds = sorted({float(p.get("score", 0.0)) for p in preds_sorted}, reverse=True)
        last_p = last_r = 0.0
        curve_precisions: List[float] = []
        curve_recalls: List[float] = []
        for thr in thresholds:
            preds_thr = [p for p in preds_sorted if float(p.get("score", 0.0)) >= thr]
            m_thr, up_thr, ug_thr = greedy_match(
                preds_thr, gts, iou_threshold=iou_threshold, require_label_match=require_label_match
            )
            tp_thr = len(m_thr)
            fp_thr = len(up_thr)
            fn_thr = len(ug_thr)
            pr = precision_recall_f1(tp_thr, fp_thr, fn_thr)
            curve_precisions.append(pr["precision"]) 
            curve_recalls.append(pr["recall"]) 
        # Interpolated AP (area under precision envelope)
        # Sort by recall ascending
        paired = sorted(zip(curve_recalls, curve_precisions))
        recalls_sorted = [r for r, _ in paired]
        precisions_sorted = [p for _, p in paired]
        # Make precision envelope monotonic
        for i in range(len(precisions_sorted) - 2, -1, -1):
            precisions_sorted[i] = max(precisions_sorted[i], precisions_sorted[i + 1])
        ap = 0.0
        for i in range(1, len(recalls_sorted)):
            dr = max(0.0, recalls_sorted[i] - recalls_sorted[i - 1])
            ap += precisions_sorted[i] * dr
        pr_points = list(zip(recalls_sorted, precisions_sorted))
    else:
        ap = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "mean_iou": mean_iou,
        "ap50": ap,
        "pr_curve": pr_points,
        "matches": matches,
        "unmatched_predictions": unmatched_p,
        "unmatched_ground_truth": unmatched_g,
    }


def parse_coco_annotations(
    coco_bytes: bytes,
    image_filename: Optional[str] = None,
) -> List[Dict]:
    """
    Parses a minimal COCO-like JSON into a list of ground truth boxes.
    Supports keys: images (with id, file_name), annotations (with image_id, bbox, category_id), categories (with id, name)
    If images is missing or contains a single image, uses all annotations.
    If multiple images and image_filename provided, filters by matching file_name.
    Returns: [{bbox:[x1,y1,x2,y2], label:str}]
    """
    data = json.loads(coco_bytes.decode("utf-8"))
    categories = data.get("categories", [])
    cat_map = {int(c["id"]): c.get("name", str(c["id"])) for c in categories if "id" in c}

    images = data.get("images")
    anns = data.get("annotations")

    gts: List[Dict] = []
    if isinstance(anns, list) and len(anns) > 0:
        image_id_filter: Optional[int] = None
        if isinstance(images, list) and len(images) > 0:
            if image_filename is not None:
                for img in images:
                    if str(img.get("file_name")) == str(image_filename):
                        image_id_filter = int(img.get("id")) if "id" in img else None
                        break
            # If still None and there is only one image, use that id
            if image_id_filter is None and len(images) == 1 and "id" in images[0]:
                image_id_filter = int(images[0]["id"])  

        for ann in anns:
            if not isinstance(ann, dict):
                continue
            if image_id_filter is not None and int(ann.get("image_id", -1)) != image_id_filter:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            xyxy = xywh_to_xyxy([float(b) for b in bbox])
            cat_id = ann.get("category_id")
            label = cat_map.get(int(cat_id), str(cat_id)) if cat_id is not None else str(ann.get("category", "unknown"))
            gts.append({"bbox": xyxy, "label": str(label)})
    else:
        # Fallback: support a very simple custom format {"annotations": [{"bbox":[x1,y1,x2,y2], "label":"Text"}, ...]}
        simple = data.get("annotations", []) if isinstance(data, dict) else []
        for ann in simple:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                gts.append({"bbox": [float(v) for v in bbox], "label": str(ann.get("label", "unknown"))})

    return gts


