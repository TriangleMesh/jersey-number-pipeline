# tracklet_consolidation.py
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any

DIGITS_1_2 = re.compile(r"^\d{1,2}$")

def normalize_pred(s: Any) -> str | None:
    """
    Normalize common STR/OCR mistakes and keep only jersey numbers 0..99.
    """
    if s is None:
        return None
    s = str(s).strip().replace(" ", "")

    # common confusions
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1").replace("|", "1")

    if not DIGITS_1_2.match(s):
        return None
    v = int(s)
    return str(v) if 0 <= v <= 99 else None

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def tracklet_id_from_image_path(img_path: str) -> str:
    """
    SoccerNet crops are typically: .../crops_folder/imgs/<TRACKLET_ID>/<img>.jpg
    So parent folder name is the tracklet id.
    """
    return os.path.basename(os.path.dirname(img_path))

def iter_rows(data: Any):
    """
    Handle common STR result formats:
    - list of dicts: [{"img":..., "pred":..., "conf":...}, ...]
    - dict keyed by path: {".../img.jpg": {"pred":..., "conf":...}, ...}
    """
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
        return

    if isinstance(data, dict):
        for k, v in data.items():
            # k might be the image path
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("img", k)
                yield row
            else:
                # value might be pred string
                yield {"img": k, "pred": v, "conf": 0.0}
        return

    raise ValueError(f"Unknown STR result format: {type(data)}")

def consolidate_tracklet(frame_preds: list[Any], frame_confs: list[Any]):
    """
    Conservative consolidation that is unlikely to reduce accuracy:
    - Uses all valid predictions (no high min_conf gate by default)
    - Weights by confidence if present, otherwise uses 1.0
    - Returns -1 only if *no* valid 0..99 predictions exist
    """
    scores = defaultdict(float)
    valid = 0

    # if confs missing or all zeros, treat as equal weights
    confs = [safe_float(c, 0.0) for c in frame_confs]
    all_zero = all(c <= 0.0 for c in confs)

    for p, c in zip(frame_preds, confs):
        p2 = normalize_pred(p)
        if p2 is None:
            continue

        # weight: use confidence if meaningful, else equal weight
        w = 1.0 if all_zero else max(c, 1e-6)

        # sum weights (simple + stable). log weighting is optional; sum is safest.
        scores[p2] += w
        valid += 1

    if valid == 0 or not scores:
        return -1, {"reason": "no_valid_digits", "valid": 0, "scores": {}}

    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])

    # mild ambiguity handling (won't cause lots of -1s)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    return int(best_label), {
        "reason": "ok",
        "valid": valid,
        "best_score": best_score,
        "margin": margin,
        "scores": dict(scores),
    }

def process_jersey_id_predictions_v2(str_result_file: str):
    """
    Read STR results, group by tracklet, consolidate, return:
      results_dict: {tracklet_id: final_pred_int}
      analysis: {tracklet_id: debug_info}
    """
    with open(str_result_file, "r") as f:
        data = json.load(f)

    groups_preds = defaultdict(list)
    groups_confs = defaultdict(list)

    for row in iter_rows(data):
        img  = row.get("img") or row.get("image") or row.get("path")
        pred = row.get("pred") or row.get("text") or row.get("prediction")
        conf = row.get("conf") or row.get("prob") or row.get("confidence") or 0.0

        if not img:
            continue

        tid = tracklet_id_from_image_path(img)
        groups_preds[tid].append(pred)
        groups_confs[tid].append(conf)

    results = {}
    analysis = {}

    for tid in groups_preds:
        final_pred, info = consolidate_tracklet(groups_preds[tid], groups_confs[tid])
        results[tid] = final_pred
        analysis[tid] = info

    return results, analysis