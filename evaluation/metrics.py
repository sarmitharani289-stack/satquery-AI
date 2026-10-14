"""
evaluation/metrics.py
=======================
Metric implementations for evaluating SatQuery AI against RSVQA,
VRSBench, CDVQA, and the ISRO/SAC Cartosat-2S/RISAT evaluation set.

All metrics are implemented from scratch with only `numpy`/stdlib
dependencies (no `nltk`/`pycocoevalcap` requirement) so the evaluator
runs in the same dependency-light environment as the rest of the
pipeline.

Implemented
-----------
* `bleu_n`            — BLEU-1..4 with brevity penalty (captioning/VQA text).
* `simple_cider`       — TF-IDF-weighted n-gram cosine similarity, a
                          lightweight CIDEr analogue (captioning).
* `iou`                — Intersection-over-Union for bounding boxes and
                          binary masks (grounding, change detection).
* `f1_precision_recall`— for binary change masks / detection sets.
* `overall_accuracy`   — exact-match accuracy (VQA / change-VQA).
* `normalize_scores`   — min-max normalisation across metric families
                          before combination, per the problem statement's
                          "Scores will be normalised before combining
                          different metrics."
* `aggregate_scores`   — weighted combination into a single leaderboard score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Text tokenisation
# --------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# --------------------------------------------------------------------------
# BLEU
# --------------------------------------------------------------------------
def bleu_n(candidate: str, references: Sequence[str], max_n: int = 4) -> float:
    """Corpus-agnostic sentence-level BLEU-`max_n` with brevity penalty."""
    cand_tokens = _tokenize(candidate)
    ref_tokens_list = [_tokenize(r) for r in references]
    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts: Counter = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = _ngrams(ref_tokens, n)
            for gram, cnt in ref_ngrams.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], cnt)
        clipped = sum(min(cnt, max_ref_counts.get(gram, 0)) for gram, cnt in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    if any(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    cand_len = len(cand_tokens)
    ref_len = min((len(r) for r in ref_tokens_list), key=lambda L: abs(L - cand_len)) \
        if ref_tokens_list else cand_len
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    return float(bp * geo_mean)


# --------------------------------------------------------------------------
# Lightweight CIDEr analogue
# --------------------------------------------------------------------------
def simple_cider(candidate: str, references: Sequence[str], n: int = 4) -> float:
    """
    TF-IDF weighted n-gram cosine similarity between candidate and each
    reference, averaged — captures the spirit of CIDEr (consensus,
    down-weighting common n-grams) without requiring a corpus-level IDF
    table (approximated per-call from the reference set itself).
    """
    cand_tokens = _tokenize(candidate)
    if not cand_tokens or not references:
        return 0.0

    scores = []
    for order in range(1, n + 1):
        cand_grams = _ngrams(cand_tokens, order)
        doc_freq: Counter = Counter()
        ref_grams_list = []
        for ref in references:
            rg = _ngrams(_tokenize(ref), order)
            ref_grams_list.append(rg)
            for g in rg:
                doc_freq[g] += 1

        def _tfidf_vec(grams: Counter) -> Dict[Tuple[str, ...], float]:
            total = sum(grams.values()) or 1
            vec = {}
            for g, c in grams.items():
                tf = c / total
                idf = math.log((len(references) + 1) / (doc_freq.get(g, 0) + 1)) + 1.0
                vec[g] = tf * idf
            return vec

        cand_vec = _tfidf_vec(cand_grams)
        for rg in ref_grams_list:
            ref_vec = _tfidf_vec(rg)
            all_keys = set(cand_vec) | set(ref_vec)
            dot = sum(cand_vec.get(k, 0.0) * ref_vec.get(k, 0.0) for k in all_keys)
            norm_c = math.sqrt(sum(v * v for v in cand_vec.values())) or 1e-9
            norm_r = math.sqrt(sum(v * v for v in ref_vec.values())) or 1e-9
            scores.append(dot / (norm_c * norm_r))
    return float(np.mean(scores)) * 10.0 if scores else 0.0  # scaled ~CIDEr range


# --------------------------------------------------------------------------
# Spatial metrics
# --------------------------------------------------------------------------
def iou_boxes(box_a: dict, box_b: dict) -> float:
    """IoU between two {"xmin","ymin","xmax","ymax"} boxes."""
    xa1, ya1, xa2, ya2 = box_a["xmin"], box_a["ymin"], box_a["xmax"], box_a["ymax"]
    xb1, yb1, xb2, yb2 = box_b["xmin"], box_b["ymin"], box_b["xmax"], box_b["ymax"]
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two boolean masks of identical shape."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def best_box_match_map(pred_boxes: List[dict], gt_boxes: List[dict],
                        iou_threshold: float = 0.5) -> float:
    """Simple mean-average-precision proxy at a single IoU threshold:
    fraction of ground-truth boxes matched by at least one predicted box
    with IoU >= threshold (recall-oriented, appropriate for grounding
    evaluation where a single correct localisation per query matters)."""
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    matched = 0
    for gt in gt_boxes:
        if any(iou_boxes(gt, pb) >= iou_threshold for pb in pred_boxes):
            matched += 1
    return matched / len(gt_boxes)


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------
def f1_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(bool).ravel()
    gt = gt_mask.astype(bool).ravel()
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def overall_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact-match accuracy after light normalisation (lower-case, strip
    punctuation) — standard for RSVQA/CDVQA-style closed-vocabulary VQA."""
    if not predictions:
        return 0.0
    correct = 0
    for p, r in zip(predictions, references):
        if _normalize_answer(p) == _normalize_answer(r):
            correct += 1
    return correct / len(predictions)


def _normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# --------------------------------------------------------------------------
# Score normalisation & aggregation
# --------------------------------------------------------------------------
def normalize_scores(raw_scores: Dict[str, float],
                      score_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Min-max normalise each named score into [0, 1] using an explicit
    (min, max) range per metric family (e.g. BLEU in [0,1], CIDEr-like
    in [0,10], IoU in [0,1], OA in [0,1]) — required because raw metric
    scales differ before they can be combined into one leaderboard
    number, per the evaluation criteria.
    """
    normed = {}
    for key, val in raw_scores.items():
        lo, hi = score_ranges.get(key, (0.0, 1.0))
        normed[key] = float(max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9))))
    return normed


def aggregate_scores(normalized_scores: Dict[str, float],
                      weights: Dict[str, float] = None) -> float:
    """Weighted mean of normalised scores; equal weighting if none given."""
    if not normalized_scores:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in normalized_scores}
    total_w = sum(weights.get(k, 0.0) for k in normalized_scores) or 1.0
    return float(sum(normalized_scores[k] * weights.get(k, 0.0) for k in normalized_scores) / total_w)
