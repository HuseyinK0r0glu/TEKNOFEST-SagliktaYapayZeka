"""JSON assembly + validation against the delivery guide's rules (section 3)."""
from __future__ import annotations

import json

from pathugen.config import (
    APPLICATION_ID, COMPETITION_LEVEL, PANELS, PROB_TOLERANCE, TEAM_ID, TEAM_NAME,
)


def build_result_json(predictions_by_panel: dict[str, list[dict]]) -> dict:
    predictions = []
    for panel in PANELS:
        predictions.extend(predictions_by_panel.get(panel, []))
    return {
        "team_name": TEAM_NAME, "team_id": TEAM_ID, "application_id": APPLICATION_ID,
        "competition_level": COMPETITION_LEVEL, "predictions": predictions,
    }


def validate_result_json(result: dict, expected_counts: dict[str, int]) -> list[str]:
    """Structural checks mirroring section 3 of the delivery guide. Returns a
    list of problems (empty = valid). This does NOT touch the network or the
    upload system — it is a local pre-flight check only."""
    problems = []
    for field in ("team_name", "team_id", "application_id", "competition_level"):
        if not result.get(field):
            problems.append(f"missing or empty top-level field: {field}")
    if result.get("competition_level") != COMPETITION_LEVEL:
        problems.append(f"competition_level must be '{COMPETITION_LEVEL}'")

    preds = result.get("predictions")
    if not isinstance(preds, list) or not preds:
        problems.append("predictions must be a non-empty array")
        return problems

    seen = {}
    for i, rec in enumerate(preds):
        where = f"predictions[{i}] (id={rec.get('id')!r}, panel={rec.get('panel')!r})"
        panel = rec.get("panel")
        if panel not in PANELS:
            problems.append(f"{where}: invalid panel")
        rid = rec.get("id")
        if not rid:
            problems.append(f"{where}: empty id")
        key = (panel, rid)
        if key in seen:
            problems.append(f"{where}: duplicate id within panel {panel}")
        seen[key] = True
        if rec.get("predicted_class") not in ("0", "1"):
            problems.append(f"{where}: predicted_class must be the string '0' or '1', "
                            f"got {rec.get('predicted_class')!r}")
        prob = rec.get("predicted_prob")
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) \
                or not (0.0 - PROB_TOLERANCE <= prob <= 1.0 + PROB_TOLERANCE) \
                or prob != prob:                      # NaN check
            problems.append(f"{where}: predicted_prob must be a finite number in "
                            f"[0, 1], got {prob!r}")

    for panel, expected_n in expected_counts.items():
        got = sum(1 for p, _ in seen if p == panel)
        if got != expected_n:
            problems.append(f"panel {panel}: expected exactly {expected_n} predictions "
                            f"(one per input row), got {got}")

    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        problems.append(f"result is not valid strict JSON (NaN/Infinity present?): {exc}")
    return problems
