"""Central configuration for model maturity tiers and per-user threshold management.

All transition day-count gates live here — nothing else in the codebase should
hardcode day thresholds.

Tier order:  rules → copula → ml_model → nn

Users advance tiers manually (UI button) or can be recommended to do so when they
qualify.  Each user can override the default thresholds via maturity_overrides JSON
stored in user_profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Default thresholds (all durations in paired-data days, not calendar days)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "copula_min_days":      15,    # ≥ 15 paired days → fit Gaussian copula
    "ml_model_min_days":    30,    # ≥ 30 paired days → fit per-outcome Ridge models
    "ml_lag2_min_days":     60,    # ≥ 60 paired days → add rolling aggregates
    "nn_min_days":          90,    # ≥ 90 paired days → train outcome NN
    "nn_accuracy_gain_pct": 10.0,  # NN must beat ML by >10% MAE to recommend transition
    "nn_consecutive_wins":  3,     # Consecutive retraining wins before auto-recommending NN
}

TIER_ORDER: list[str] = ["rules", "copula", "ml_model", "nn"]

TIER_META: dict[str, dict[str, str]] = {
    "rules": {
        "label": "Rule-Based",
        "color": "#6b7280",
        "desc": "Hand-tuned physiological rules power predictions.",
    },
    "copula": {
        "label": "Copula",
        "color": "#f59e0b",
        "desc": "Gaussian copula fitted on your data captures your personal input→outcome distribution.",
    },
    "ml_model": {
        "label": "ML Model",
        "color": "#8b5cf6",
        "desc": "Ridge regression per outcome trained on your history with R² diagnostics.",
    },
    "nn": {
        "label": "Neural Network",
        "color": "#22c55e",
        "desc": "Neural network trained on real + synthetic multi-lag data. Highest personalization.",
    },
}


# ---------------------------------------------------------------------------
# MaturityStatus dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaturityStatus:
    data_days: int
    active_tier: str
    next_tier: str | None
    prev_tier: str | None
    qualifies_for_next: bool
    recommendation_text: str
    can_advance: bool
    can_revert: bool
    thresholds: dict[str, Any]
    tier_meta: dict[str, dict[str, str]]
    nn_winning: bool = False

    def to_dict(self) -> dict:
        return {
            "data_days": self.data_days,
            "active_tier": self.active_tier,
            "active_tier_label": TIER_META[self.active_tier]["label"],
            "active_tier_color": TIER_META[self.active_tier]["color"],
            "active_tier_desc":  TIER_META[self.active_tier]["desc"],
            "next_tier": self.next_tier,
            "next_tier_label": TIER_META[self.next_tier]["label"] if self.next_tier else None,
            "prev_tier": self.prev_tier,
            "prev_tier_label": TIER_META[self.prev_tier]["label"] if self.prev_tier else None,
            "qualifies_for_next": self.qualifies_for_next,
            "recommendation_text": self.recommendation_text,
            "can_advance": self.can_advance,
            "can_revert": self.can_revert,
            "thresholds": self.thresholds,
            "tier_meta": self.tier_meta,
            "nn_winning": self.nn_winning,
        }


# ---------------------------------------------------------------------------
# Per-user override helpers
# ---------------------------------------------------------------------------

def _load_overrides(user_id: int) -> dict:
    """Load raw maturity_overrides JSON from DB for a user."""
    from backend.database import SessionLocal, UserProfile
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile or not getattr(profile, "maturity_overrides", None):
            return {}
        raw = profile.maturity_overrides
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return {}
    finally:
        db.close()


def _save_overrides(user_id: int, overrides: dict) -> None:
    """Persist maturity_overrides JSON to DB."""
    from backend.database import SessionLocal, UserProfile
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile is None:
            return
        profile.maturity_overrides = json.dumps(overrides)
        db.commit()
    finally:
        db.close()


def get_user_thresholds(user_id: int) -> dict[str, Any]:
    """Return effective thresholds: defaults merged with per-user overrides."""
    overrides = _load_overrides(user_id)
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(overrides.get("thresholds", {}))
    return merged


def set_user_thresholds(user_id: int, threshold_overrides: dict[str, Any]) -> None:
    """Save per-user threshold overrides. Only keys present in DEFAULT_THRESHOLDS are accepted."""
    valid = {k: v for k, v in threshold_overrides.items() if k in DEFAULT_THRESHOLDS}
    overrides = _load_overrides(user_id)
    overrides["thresholds"] = {**overrides.get("thresholds", {}), **valid}
    _save_overrides(user_id, overrides)


def get_active_tier(user_id: int) -> str:
    """Return the tier the user is currently using (may be behind their eligible tier)."""
    overrides = _load_overrides(user_id)
    tier = overrides.get("active_tier")
    if tier and tier in TIER_ORDER:
        return tier
    # First time: default to max eligible so existing users are not demoted
    return _max_eligible_tier(user_id)


def set_active_tier(user_id: int, tier: str) -> None:
    """Persist the user's active tier."""
    if tier not in TIER_ORDER:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {TIER_ORDER}")
    overrides = _load_overrides(user_id)
    overrides["active_tier"] = tier
    _save_overrides(user_id, overrides)


# ---------------------------------------------------------------------------
# Data-day counting
# ---------------------------------------------------------------------------

def count_paired_days(user_id: int) -> int:
    """Count valid action@T → outcome@T+1 sequential day pairs.

    A pair is valid when both sync_date=T and sync_date=T+1 exist in
    wearable_syncs and T+1 is not today (still accumulating).
    """
    from backend.database import SessionLocal, WearableSync
    today_str = date.today().isoformat()
    db = SessionLocal()
    try:
        rows = db.query(WearableSync.sync_date).filter(
            WearableSync.user_id == user_id
        ).all()
        dates = sorted({r.sync_date for r in rows})
        count = 0
        for i in range(len(dates) - 1):
            t0, t1 = dates[i], dates[i + 1]
            if t1 == today_str:
                break
            # Only count consecutive days (not gaps)
            from datetime import datetime as _dt
            try:
                d0 = _dt.strptime(t0, "%Y-%m-%d").date()
                d1 = _dt.strptime(t1, "%Y-%m-%d").date()
                if (d1 - d0).days == 1:
                    count += 1
            except Exception:
                continue
        return count
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tier eligibility
# ---------------------------------------------------------------------------

def _max_eligible_tier(user_id: int) -> str:
    """Compute the highest tier the user qualifies for (data + model file needed)."""
    import os
    thresholds = get_user_thresholds(user_id)
    days = count_paired_days(user_id)

    models_root = os.path.join(os.path.dirname(__file__), "..", "models")
    nn_path   = os.path.join(models_root, f"user_{user_id}", "outcome_nn.pt")
    ml_path   = os.path.join(models_root, f"user_{user_id}", "outcome_models.json")
    dist_path = os.path.join(models_root, f"user_{user_id}", "distribution.json")

    if days >= thresholds["nn_min_days"] and os.path.exists(nn_path):
        return "nn"
    if days >= thresholds["ml_model_min_days"] and os.path.exists(ml_path):
        return "ml_model"
    if days >= thresholds["copula_min_days"] and os.path.exists(dist_path):
        return "copula"
    return "rules"


def get_maturity_status(user_id: int) -> MaturityStatus:
    """Compute full maturity status for a user."""
    thresholds = get_user_thresholds(user_id)
    days = count_paired_days(user_id)
    active_tier = get_active_tier(user_id)
    eligible_tier = _max_eligible_tier(user_id)

    tier_idx = TIER_ORDER.index(active_tier)
    eligible_idx = TIER_ORDER.index(eligible_tier)

    next_tier = TIER_ORDER[tier_idx + 1] if tier_idx < len(TIER_ORDER) - 1 else None
    prev_tier = TIER_ORDER[tier_idx - 1] if tier_idx > 0 else None

    # Map each current tier's minimum threshold key for the NEXT tier
    _next_threshold_key: dict[str, str | None] = {
        "rules":    "copula_min_days",
        "copula":   "ml_model_min_days",
        "ml_model": "nn_min_days",
        "nn":       None,
    }
    next_key = _next_threshold_key[active_tier]
    next_min = thresholds.get(next_key, 0) if next_key else None

    # User qualifies for next tier if their eligible tier is ahead of their active tier
    qualifies_for_next = (next_tier is not None) and (eligible_idx > tier_idx)

    # NN winning flag
    overrides = _load_overrides(user_id)
    nn_comparison = overrides.get("nn_comparison", {})
    nn_winning = nn_comparison.get("nn_winning", False)

    # Human-readable recommendation
    if active_tier == "nn":
        rec_text = "You are on the most advanced tier. Neural Network is fully active."
    elif qualifies_for_next:
        rec_text = (
            f"You qualify for {TIER_META[next_tier]['label']}! "
            f"Advance when ready to unlock more accurate predictions."
        )
    elif next_min is not None and next_tier is not None:
        days_remaining = max(0, next_min - days)
        rec_text = (
            f"{days_remaining} more day{'s' if days_remaining != 1 else ''} of data needed "
            f"to unlock {TIER_META[next_tier]['label']}."
        )
    else:
        rec_text = "Collecting data for next tier."

    return MaturityStatus(
        data_days=days,
        active_tier=active_tier,
        next_tier=next_tier,
        prev_tier=prev_tier,
        qualifies_for_next=qualifies_for_next,
        recommendation_text=rec_text,
        can_advance=qualifies_for_next and next_tier is not None,
        can_revert=prev_tier is not None,
        thresholds=thresholds,
        tier_meta=TIER_META,
        nn_winning=nn_winning,
    )


# ---------------------------------------------------------------------------
# NN vs ML comparison tracking
# ---------------------------------------------------------------------------

def record_nn_comparison(user_id: int, nn_mae: float, ml_mae: float) -> None:
    """Store NN vs ML accuracy comparison and update consecutive-win counter."""
    thresholds = get_user_thresholds(user_id)
    gain_needed = thresholds["nn_accuracy_gain_pct"] / 100.0
    win = (ml_mae > 0) and ((ml_mae - nn_mae) / ml_mae >= gain_needed)

    overrides = _load_overrides(user_id)
    comp = overrides.get("nn_comparison", {"wins": 0, "total": 0})
    comp["nn_mae"] = nn_mae
    comp["ml_mae"] = ml_mae
    comp["total"] = comp.get("total", 0) + 1
    if win:
        comp["wins"] = comp.get("wins", 0) + 1
    else:
        comp["wins"] = 0  # reset consecutive-win streak
    comp["nn_winning"] = comp["wins"] >= thresholds["nn_consecutive_wins"]
    overrides["nn_comparison"] = comp
    _save_overrides(user_id, overrides)
