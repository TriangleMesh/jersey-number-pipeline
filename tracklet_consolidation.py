from collections import defaultdict
import math
import os
import re
import json

DIGITS = re.compile(r"^\d{1,2}$")

def normalize_pred(s: str):
    if s is None:
        return None
    s = str(s).strip().replace(" ", "")
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1").replace("|", "1")
    if not DIGITS.match(s):
        return None
    v = int(s)
    return str(v) if 0 <= v <= 99 else None

def consolidate_tracklet(frame_preds, frame_confs,
                         min_conf=0.45,
                         min_valid_frames=2):
    scores = defaultdict(float)
    valid = 0
    for p, c in zip(frame_preds, frame_confs):
        try:
            c = float(c)
        except:
            c = 0.0
        if c < min_conf:
            continue
        p2 = normalize_pred(p)
        if p2 is None:
            continue
        scores[p2] += math.log(max(c, 1e-6))
        valid += 1

    if valid < min_valid_frames or not scores:
        return -1, {"reason": "too_few_valid_frames", "valid": valid, "scores": dict(scores)}

    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return int(best_label), {"reason": "ok", "valid": valid, "scores": dict(scores)}

def _tracklet_id_from_path(img_path: str) -> str:
    return os.path.basename(os.path.dirname(img_path))

def process_jersey_id_predictions_v2(str_result_file: str):
    with open(str_result_file, "r") as f:
        data = json.load(f)

    groups_preds = defaultdict(list)
    groups_confs = defaultdict(list)

    # Your tmp_str_results.json format is a LIST of dicts: {"img","pred","conf"}
    for row in data:
        img = row.get("img")
        pred = row.get("pred")
        conf = row.get("conf", 0.0)
        tid = _tracklet_id_from_path(img)
        groups_preds[tid].append(pred)
        groups_confs[tid].append(conf)

    results = {}
    analysis = {}
    for tid in groups_preds:
        final_pred, info = consolidate_tracklet(groups_preds[tid], groups_confs[tid])
        results[tid] = final_pred
        analysis[tid] = info

    return results, analysis
