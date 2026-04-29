"""Evaluation service for outcome ML models.

Provides API-ready payloads from the trained OutcomeModelSuite:
  - get_outcome_model_evals()  — R², importances, explanations, compliance summary
  - get_model_trajectory()     — training history for the R² trend chart
  - get_inference_comparison() — per-path rolling MAE comparison (ML vs copula)

Called from app.py endpoints:
  GET /api/evals/models      → get_outcome_model_evals(user_id)
  GET /api/evals/trajectory  → get_model_trajectory(user_id)
"""

from __future__ import annotations

from typing import Any

from backend.outcome_models import get_eval_payload, load_trajectory
from backend.maturity_config import get_maturity_status


def get_outcome_model_evals(user_id: int) -> dict[str, Any]:
    """Full evals payload: maturity status + model cards + compliance summary."""
    status = get_maturity_status(user_id)
    payload = get_eval_payload(user_id)
    payload["maturity"] = status.to_dict()
    return payload


def get_model_trajectory(user_id: int) -> list[dict]:
    """Load the R² trajectory JSONL for the trend chart."""
    return load_trajectory(user_id)


def get_inference_comparison(user_id: int) -> dict[str, Any]:
    """Compute rolling 7-day MAE per inference path (primary vs alt).

    Returns a dict with 'primary_mae', 'alt_mae', 'rec_count', and
    'transition_recommended' (True if alt path qualifies for promotion).
    """
    from backend.database import SessionLocal, Recommendation
    import json

    db = SessionLocal()
    try:
        recs = (
            db.query(Recommendation)
            .filter(Recommendation.user_id == user_id)
            .filter(Recommendation.fidelity_score.isnot(None))
            .order_by(Recommendation.rec_date.desc())
            .limit(7)
            .all()
        )

        if not recs:
            return {
                "available": False,
                "reason": "No evaluated recommendations found.",
            }

        primary_scores = []
        alt_scores = []

        for r in recs:
            if r.fidelity_score is not None:
                primary_scores.append(float(r.fidelity_score))
            if r.fidelity_score_alt is not None:
                alt_scores.append(float(r.fidelity_score_alt))

        primary_mae = 1.0 - (sum(primary_scores) / len(primary_scores)) if primary_scores else None
        alt_mae = 1.0 - (sum(alt_scores) / len(alt_scores)) if alt_scores else None

        transition_recommended = False
        if primary_mae is not None and alt_mae is not None:
            from backend.maturity_config import get_user_thresholds
            thresholds = get_user_thresholds(user_id)
            gain_pct = thresholds["nn_accuracy_gain_pct"]
            if primary_mae > 0:
                improvement = (primary_mae - alt_mae) / primary_mae * 100
                transition_recommended = improvement >= gain_pct

        return {
            "available": True,
            "primary_path": recs[0].inference_path if recs else None,
            "primary_mae": round(primary_mae, 4) if primary_mae is not None else None,
            "alt_mae": round(alt_mae, 4) if alt_mae is not None else None,
            "rec_count": len(recs),
            "transition_recommended": transition_recommended,
        }

    finally:
        db.close()
