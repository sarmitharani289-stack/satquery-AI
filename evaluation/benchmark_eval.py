"""
evaluation/benchmark_eval.py
===============================
Evaluation harness that runs the full SatQuery AI agentic pipeline
(`agent.orchestrator.AgentController`) over prescribed public benchmark
test splits (VRSBench, RSVQA, CDVQA) and the ISRO/SAC Cartosat-2S/RISAT
evaluation set, then scores the outputs with `evaluation.metrics`.

Expected annotation format
----------------------------
Each benchmark is described by a JSON-lines (`.jsonl`) or JSON list
annotation file with one of the following per-record schemas
(auto-detected by `task` field):

VQA / Change-VQA record:
    {"task": "vqa" | "change_vqa", "images": ["t1.tif", "t2.tif"?],
     "query": "...", "answer": "..."}

Captioning record:
    {"task": "caption", "images": ["img.tif"], "references": ["...", "..."]}

Grounding record:
    {"task": "grounding", "images": ["img.tif"], "query": "...",
     "boxes": [{"xmin":.., "ymin":.., "xmax":.., "ymax":..}]}

Change-detection record:
    {"task": "change_detection", "images": ["t1.tif", "t2.tif"],
     "query": "...", "change_mask_path": "mask.png"}  (mask: nonzero = changed)

This module does not assume the annotation files are present in this
repository (they are large, license-restricted benchmark downloads);
`load_annotations` raises a clear `FileNotFoundError` if the path does
not exist, and `run_benchmark` can be pointed at any directory that
follows the schema above, including a held-out ISRO/SAC evaluation set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from agent.orchestrator import AgentController
from core import config
from evaluation import metrics as M

SCORE_RANGES = {
    "bleu4": (0.0, 1.0),
    "cider": (0.0, 10.0),
    "oa": (0.0, 1.0),
    "grounding_recall@0.5": (0.0, 1.0),
    "change_f1": (0.0, 1.0),
}

DEFAULT_WEIGHTS = {
    "bleu4": 1.0, "cider": 1.0, "oa": 1.5,
    "grounding_recall@0.5": 1.5, "change_f1": 1.5,
}


def load_annotations(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Benchmark annotation file not found: {path}. Point this at the "
            f"prescribed test-split JSON/JSONL for VRSBench/RSVQA/CDVQA or the "
            f"ISRO/SAC evaluation set (annotations are not bundled with this repo).")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def run_benchmark(annotation_path: str, image_root: str, dataset_name: str = "benchmark") -> dict:
    """
    Run the full agentic pipeline over every record in `annotation_path`
    and compute per-task and aggregate normalised metrics.

    Returns
    -------
    dict with keys: "dataset", "n_records", "per_task_raw", "per_task_normalized",
    "aggregate_score", "per_record_results".
    """
    records = load_annotations(annotation_path)
    controller = AgentController()
    root = Path(image_root)

    raw_scores: Dict[str, List[float]] = {k: [] for k in SCORE_RANGES}
    per_record_results = []

    for rec in records:
        task = rec["task"]
        image_paths = [str(root / p) for p in rec["images"]]
        query = rec.get("query", "Describe the land-cover and major objects visible in this image.")

        try:
            trace = controller.run_query(image_paths, query)
        except Exception as exc:
            per_record_results.append({"record": rec, "error": str(exc)})
            continue

        answer = trace.get("evidence_grounded_answer", "") or ""
        pred_boxes = trace["spatial_evidence"]["bounding_boxes"]

        record_result = {"task": task, "trace_status": trace["status"], "answer": answer}

        if task == "caption" and "references" in rec:
            bleu = M.bleu_n(answer, rec["references"], max_n=4)
            cider = M.simple_cider(answer, rec["references"])
            raw_scores["bleu4"].append(bleu)
            raw_scores["cider"].append(cider)
            record_result.update({"bleu4": bleu, "cider": cider})

        elif task in ("vqa", "change_vqa") and "answer" in rec:
            oa = M.overall_accuracy([answer], [rec["answer"]])
            raw_scores["oa"].append(oa)
            record_result["oa"] = oa

        elif task == "grounding" and "boxes" in rec:
            recall = M.best_box_match_map(pred_boxes, rec["boxes"], iou_threshold=0.5)
            raw_scores["grounding_recall@0.5"].append(recall)
            record_result["grounding_recall@0.5"] = recall

        elif task == "change_detection" and "change_mask_path" in rec:
            gt_mask = _load_mask(str(root / rec["change_mask_path"]))
            pred_masks = [m for m in trace["spatial_evidence"]["masks"] if m.get("label") == "change_mask"]
            if pred_masks and gt_mask is not None:
                # Trace only stores mask summaries (area), not raw arrays,
                # by design (JSON-serialisable trace); for pixel-level F1
                # scoring, re-run change_analyzer directly when precise
                # arrays are required — see `evaluate_change_detection_direct`.
                record_result["note"] = ("Pixel-level change-mask F1 requires direct module "
                                          "invocation; see evaluate_change_detection_direct().")

        per_record_results.append(record_result)

    per_task_raw = {k: float(np.mean(v)) if v else None for k, v in raw_scores.items()}
    available = {k: v for k, v in per_task_raw.items() if v is not None}
    normalized = M.normalize_scores(available, SCORE_RANGES)
    aggregate = M.aggregate_scores(normalized, DEFAULT_WEIGHTS)

    return {
        "dataset": dataset_name,
        "n_records": len(records),
        "per_task_raw": per_task_raw,
        "per_task_normalized": normalized,
        "aggregate_score": aggregate,
        "per_record_results": per_record_results,
    }


def evaluate_change_detection_direct(annotation_path: str, image_root: str) -> dict:
    """
    Pixel-level change-detection F1 evaluation that bypasses the JSON
    trace (which intentionally omits raw mask arrays) by calling
    `models.change_analyzer` directly, for CDVQA-style mask-level
    scoring against reference change masks.
    """
    from models.change_analyzer import run as change_run
    from core.geo_io import read_image
    from core.validator import classify_modality

    records = [r for r in load_annotations(annotation_path) if r["task"] == "change_detection"]
    root = Path(image_root)
    f1_scores = []

    for rec in records:
        t1 = read_image(str(root / rec["images"][0]))
        t2 = read_image(str(root / rec["images"][1]))
        t1.modality = classify_modality(t1)
        t2.modality = classify_modality(t2)
        result = change_run(images=[t1, t2], query=rec.get("query", "What changed?"),
                             task_type=config.TASK_CHANGE_DETECTION)
        pred_mask = next((m["array"] for m in result.masks if m["label"] == "change_mask"), None)
        gt_mask = _load_mask(str(root / rec["change_mask_path"]))
        if pred_mask is not None and gt_mask is not None:
            f1_scores.append(M.f1_precision_recall(pred_mask, gt_mask)["f1"])

    return {
        "n_records": len(records),
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else None,
        "per_record_f1": f1_scores,
    }


def _load_mask(path: str) -> np.ndarray:
    from PIL import Image
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0
