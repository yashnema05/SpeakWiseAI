
import json
from pathlib import Path


FLAW_ALIASES = {
    "possible extended pause": "extra silence",
    "extended pause": "extra silence",
    "extra silence": "extra silence",

    "possible reduced pitch variation": "monotone speech",
    "reduced pitch variation": "monotone speech",
    "monotone speech": "monotone speech",

    "abnormal pacing": "abnormal pacing",
    "slow pacing": "abnormal pacing",
    "fast pacing": "abnormal pacing",

    "possible speech-activity difference": "speech activity difference",
    "speech-activity difference": "speech activity difference",
    "speech activity difference": "speech activity difference",
}


def normalize_flaw_name(name):
    normalized = str(name).lower().strip()
    return FLAW_ALIASES.get(normalized, normalized)


def calculate_temporal_iou(pred_start, pred_end, true_start, true_end):
    pred_start = float(pred_start)
    pred_end = float(pred_end)
    true_start = float(true_start)
    true_end = float(true_end)

    if pred_end < pred_start or true_end < true_start:
        return 0.0

    intersection = max(
        0.0,
        min(pred_end, true_end) - max(pred_start, true_start)
    )

    union = max(pred_end, true_end) - min(pred_start, true_start)

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_flaws(predictions, ground_truth, iou_threshold=0.5):
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1.")

    predictions = predictions or []
    ground_truth = ground_truth or []

    matched_truth = set()
    true_positives = 0
    false_positives = 0
    iou_scores = []
    matches = []

    for prediction in predictions:
        try:
            pred_start = float(prediction["start"])
            pred_end = float(prediction["end"])

            if pred_start < 0 or pred_end <= pred_start:
                raise ValueError

        except (KeyError, TypeError, ValueError):
            false_positives += 1
            continue

        prediction_name = normalize_flaw_name(
            prediction.get("flaw", "")
        )

        best_match = None
        best_iou = 0.0

        for index, truth in enumerate(ground_truth):
            if index in matched_truth:
                continue

            truth_name = normalize_flaw_name(
                truth.get("flaw", "")
            )

            if prediction_name != truth_name:
                continue

            try:
                true_start = float(truth["start"])
                true_end = float(truth["end"])

                if true_start < 0 or true_end <= true_start:
                    continue

            except (KeyError, TypeError, ValueError):
                continue

            iou = calculate_temporal_iou(
                pred_start,
                pred_end,
                true_start,
                true_end
            )

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = index

        if best_match is not None:
            true_positives += 1
            matched_truth.add(best_match)
            iou_scores.append(best_iou)

            truth = ground_truth[best_match]

            matches.append({
                "predicted_flaw": prediction.get("flaw"),
                "ground_truth_flaw": truth.get("flaw"),
                "predicted_start": pred_start,
                "predicted_end": pred_end,
                "ground_truth_start": float(truth["start"]),
                "ground_truth_end": float(truth["end"]),
                "predicted_severity": prediction.get("severity"),
                "ground_truth_severity": truth.get("severity"),
                "severity_correct": (
                    prediction.get("severity") == truth.get("severity")
                    if prediction.get("severity") is not None
                    else None
                ),
                "temporal_iou": round(best_iou, 4)
            })

        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_truth)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    mean_iou = (
        sum(iou_scores) / len(iou_scores)
        if iou_scores else 0.0
    )

    severity_matches = [
        match for match in matches
        if match["severity_correct"] is not None
    ]

    severity_accuracy = (
        sum(m["severity_correct"] for m in severity_matches)
        / len(severity_matches)
        if severity_matches else None
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "temporal_iou": round(mean_iou, 4),
        "severity_accuracy": (
            round(severity_accuracy, 4)
            if severity_accuracy is not None else None
        ),
        "iou_threshold": iou_threshold,
        "matches": matches
    }


def evaluate_by_category(predictions, ground_truth, iou_threshold=0.5):
    categories = sorted({
        normalize_flaw_name(item.get("flaw", ""))
        for item in (ground_truth or []) + (predictions or [])
    })

    results = {}

    for category in categories:
        category_predictions = [
            p for p in predictions or []
            if normalize_flaw_name(p.get("flaw", "")) == category
        ]

        category_truth = [
            g for g in ground_truth or []
            if normalize_flaw_name(g.get("flaw", "")) == category
        ]

        results[category] = evaluate_flaws(
            category_predictions,
            category_truth,
            iou_threshold
        )

    return results


def load_ground_truth(file_path="dataset/ground_truth.json"):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    test_cases = data.get("test_cases", [])

    if not isinstance(test_cases, list):
        raise ValueError(
            "Ground-truth JSON must contain a test_cases list."
        )

    return test_cases
