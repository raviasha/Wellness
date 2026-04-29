"""Physiological simulator — computes how actions change biomarkers.

MVP 2: Garmin-only, 5D action space (sleep_duration × bedtime × activity_level ×
exercise_type × exercise_duration), 7 biomarkers.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Any

from .models import (
    Action,
    ActivityLevel,
    ACTIVITY_INTENSITY,
    BedtimeWindow,
    BEDTIME_HOUR,
    Biomarkers,
    BiomarkerDeltas,
    ExerciseDuration,
    EXERCISE_DURATION_MINUTES,
    ExerciseType,
    EXERCISE_TYPE_PROPERTIES,
    SleepDuration,
    SLEEP_HOURS,
)
from .personas import PersonaConfig, ResponseModel


INTENSE_ACTIVITIES = {ActivityLevel.VIGOROUS_ACTIVITY, ActivityLevel.HIGH_INTENSITY}

# Overtraining risk per exercise type (relative weight)
_OVERTRAINING_RISK: dict[ExerciseType, float] = {
    ExerciseType.NONE: 0.0,
    ExerciseType.CARDIO: 0.8,
    ExerciseType.STRENGTH: 0.9,
    ExerciseType.FLEXIBILITY: 0.0,
    ExerciseType.HIIT: 1.3,
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _consecutive_intense_days(history: list[dict[str, Any]]) -> int:
    """Count consecutive intense activity days looking backward."""
    count = 0
    for h in reversed(history):
        act = h.get("actual_action", {}).get("activity", "rest_day")
        if act in {e.value for e in INTENSE_ACTIVITIES}:
            count += 1
        else:
            break
    return count


def _consecutive_hiit_days(history: list[dict[str, Any]]) -> int:
    """Count consecutive HIIT days looking backward."""
    count = 0
    for h in reversed(history):
        ex_type = h.get("actual_action", {}).get("exercise_type", ExerciseType.NONE.value)
        if ex_type == ExerciseType.HIIT.value:
            count += 1
        else:
            break
    return count


def _recent_sleep_debt(history: list[dict[str, Any]], window: int = 7) -> float:
    """Accumulated sleep debt over the last `window` days."""
    debt = 0.0
    for h in history[-window:]:
        sleep_val = h.get("actual_action", {}).get("sleep", "7_to_8h")
        hours = SLEEP_HOURS.get(SleepDuration(sleep_val), 7.5)
        if hours < 7.0:
            debt += 7.0 - hours
    return debt


def _circadian_alignment(bedtime: BedtimeWindow) -> float:
    """Return an alignment score in [0, 1].

    10–11pm = 1.0 (optimal), degrades linearly for earlier / later bedtimes.
    """
    alignment_map = {
        BedtimeWindow.EARLY: 0.75,
        BedtimeWindow.OPTIMAL: 1.00,
        BedtimeWindow.LATE: 0.60,
        BedtimeWindow.VERY_LATE: 0.30,
        BedtimeWindow.EXTREMELY_LATE: 0.10,
    }
    return alignment_map.get(bedtime, 0.6)


def _duration_factor(exercise_duration: ExerciseDuration, exercise_type: ExerciseType) -> float:
    """Return a duration amplification scalar (1.0 baseline at zero/short).

    Longer sessions amplify both benefits AND costs with diminishing returns.
    Flexibility saturates quickly; HIIT amplifies more steeply.
    """
    minutes = EXERCISE_DURATION_MINUTES[exercise_duration]
    if minutes == 0:
        return 0.0
    base = minutes / 30.0  # normalize so 30 min ≈ 1.0
    if exercise_type == ExerciseType.HIIT:
        # Steeper curve — 60 min ≈ 2.0, 75 min ≈ 2.3
        return min(2.5, base ** 1.1)
    elif exercise_type == ExerciseType.FLEXIBILITY:
        # Quick saturation — 45 min ≈ 1.4
        return min(1.5, base ** 0.7)
    else:
        # Standard curve — 60 min ≈ 2.0, 75 min ≈ 2.2
        return min(2.2, base ** 0.9)


def compute_biomarker_changes(
    action: Action,
    current: Biomarkers,
    persona: PersonaConfig,
    history: list[dict[str, Any]],
    rng: stdlib_random.Random,
) -> BiomarkerDeltas:
    """Compute the change in each biomarker for one day.

    5D action: sleep_duration × bedtime × activity_level × exercise_type × exercise_duration.
    7 outcomes: RHR, HRV, sleep_score, stress, body_battery, sleep_stage_quality, VO2max.
    """
    rm = persona.response_model
    hours = SLEEP_HOURS[action.sleep]
    sleep_debt = _recent_sleep_debt(history)
    consecutive_intense = _consecutive_intense_days(history)
    is_intense = action.activity in INTENSE_ACTIVITIES

    exercise_type = getattr(action, "exercise_type", ExerciseType.NONE)
    exercise_duration = getattr(action, "exercise_duration", ExerciseDuration.NONE)
    bedtime = getattr(action, "bedtime", BedtimeWindow.OPTIMAL)

    ex_props = EXERCISE_TYPE_PROPERTIES[exercise_type]
    dur_factor = _duration_factor(exercise_duration, exercise_type)
    circadian = _circadian_alignment(bedtime)

    # Overtraining: type-aware risk, amplified by duration
    hiit_consecutive = _consecutive_hiit_days(history)
    type_risk = _OVERTRAINING_RISK.get(exercise_type, 0.8)
    # HIIT overtraining kicks in after 2 consecutive days; others after rm.overtraining_threshold
    if exercise_type == ExerciseType.HIIT:
        is_overtraining = is_intense and hiit_consecutive >= 2
    else:
        is_overtraining = (
            is_intense and consecutive_intense >= rm.overtraining_threshold
            and exercise_type != ExerciseType.FLEXIBILITY
        )

    # Amplify overtraining signal by duration
    overtraining_amp = 1.0 + (dur_factor * type_risk * 0.4) if is_overtraining else 0.0

    # Sleep debt penalty
    debt_factor = max(0.2, 1.0 - rm.sleep_debt_exercise_penalty * sleep_debt)

    # Circadian × sleep interaction: aligned bedtime amplifies sleep quality effects
    circadian_sleep_bonus = circadian  # 0–1 scalar

    # =====================================================================
    # Resting Heart Rate (lower is better)
    # =====================================================================
    d_rhr = 0.0
    if hours >= 7.0:
        d_rhr += rm.rhr_sleep_benefit * ((hours - 7.0) / 2.0 + 1.0) * circadian_sleep_bonus
    elif hours < 6.0:
        d_rhr += 0.3 * (1.0 + (1.0 - circadian))  # late + short sleep → bigger cost
    # Cardio / HIIT give strongest RHR benefit; strength moderate; flexibility minimal
    cardio_rhr = ex_props["cardio_benefit"] * rm.rhr_exercise_benefit * dur_factor * debt_factor
    d_rhr += cardio_rhr
    if is_overtraining:
        d_rhr += 0.5 * (1.0 + overtraining_amp)
    d_rhr += rng.gauss(0, 1.5)

    # =====================================================================
    # HRV (higher is better)
    # =====================================================================
    d_hrv = 0.0
    if hours >= 7.0:
        d_hrv += rm.hrv_sleep_sensitivity * (hours - 7.0) * circadian_sleep_bonus
    elif hours < 6.0:
        d_hrv -= rm.hrv_sleep_sensitivity * 1.5 * (1.0 + (1.0 - circadian))
    if is_overtraining:
        d_hrv -= 5.0 * (1.0 + overtraining_amp * 0.5)
    # Flexibility/yoga gives the biggest HRV boost; cardio moderate; HIIT suppresses acutely
    if exercise_type == ExerciseType.FLEXIBILITY:
        d_hrv += 3.0 * dur_factor * debt_factor
    elif exercise_type == ExerciseType.CARDIO:
        d_hrv += 1.5 * dur_factor * debt_factor
    elif exercise_type == ExerciseType.HIIT and not is_overtraining:
        d_hrv -= 1.0 * dur_factor  # acute suppression, long-term adaptation elsewhere
    elif action.activity == ActivityLevel.LIGHT_ACTIVITY and exercise_type == ExerciseType.NONE:
        d_hrv += 2.0 * debt_factor  # legacy: general light activity
    if current.stress_avg > 60:
        d_hrv -= 1.0
    d_hrv += rng.gauss(0, 5.0)

    # =====================================================================
    # Sleep Score (higher is better)
    # =====================================================================
    d_sleep = 0.0
    if hours >= 7.0 and hours <= 9.0:
        d_sleep += 1.0 * circadian_sleep_bonus
    elif hours < 6.0:
        d_sleep -= 2.0 * (1.0 + (1.0 - circadian) * 0.5)
    elif hours > 9.0:
        d_sleep -= 0.5
    # Moderate / flexibility benefits sleep; intense + late bedtime hurts it
    if exercise_type == ExerciseType.FLEXIBILITY:
        d_sleep += 1.2 * dur_factor
    elif action.activity in {ActivityLevel.MODERATE_ACTIVITY, ActivityLevel.LIGHT_ACTIVITY}:
        d_sleep += 0.8
    elif is_intense:
        late_bedtime_penalty = 1.0 - circadian  # 0 if optimal, up to 0.9 if very late
        d_sleep -= 0.5 + late_bedtime_penalty * 1.0
    if current.stress_avg > 60:
        d_sleep -= 1.0
    d_sleep += rng.gauss(0, 3.0)

    # =====================================================================
    # Stress (lower is better)
    # =====================================================================
    d_stress = 0.0
    if hours >= 7.0:
        d_stress += rm.cortisol_sleep_recovery * (hours - 7.0) / 2.0 * circadian_sleep_bonus
    elif hours < 6.0:
        d_stress += 5.0 * (1.0 + (1.0 - circadian) * 0.5)
    # Exercise type effects: flexibility best for stress; HIIT worst acutely
    cortisol = ex_props["cortisol_factor"] * rm.cortisol_exercise_stress * dur_factor
    d_stress += cortisol
    if exercise_type == ExerciseType.FLEXIBILITY:
        d_stress -= 3.0 * dur_factor  # yoga / stretching reduces stress
    elif action.activity in {ActivityLevel.LIGHT_ACTIVITY, ActivityLevel.MODERATE_ACTIVITY} and exercise_type == ExerciseType.NONE:
        d_stress -= 2.0
    elif action.activity == ActivityLevel.REST_DAY and exercise_type == ExerciseType.NONE:
        d_stress -= 1.0
    if is_overtraining:
        d_stress += rm.overtraining_cortisol_spike * (1.0 + overtraining_amp * 0.5)
    if current.stress_avg > 50:
        d_stress -= 1.0
    d_stress += rng.gauss(0, 5.0)

    # =====================================================================
    # Body Battery (higher is better)
    # =====================================================================
    d_battery = 0.0
    if hours >= 7.0 and hours <= 9.0:
        d_battery += rm.energy_sensitivity * 0.6 * circadian_sleep_bonus
    elif hours < 6.0:
        d_battery -= rm.energy_sensitivity * (1.0 + (1.0 - circadian) * 0.3)
    elif hours > 9.0:
        d_battery -= rm.energy_sensitivity * 0.3
    # Recovery cost by type and duration
    d_battery -= ex_props["recovery_cost"] * dur_factor * 3.0
    if action.activity == ActivityLevel.REST_DAY and exercise_type == ExerciseType.NONE:
        d_battery += 1.0
    elif exercise_type == ExerciseType.FLEXIBILITY:
        d_battery += 1.5 * dur_factor  # restorative
    if is_overtraining:
        d_battery -= 8.0 * (1.0 + overtraining_amp * 0.3)
    d_battery -= sleep_debt * 1.5
    if current.stress_avg > 60:
        d_battery -= 2.0
    d_battery += rng.gauss(0, 5.0)

    # =====================================================================
    # Sleep Stage Quality (deep + REM %)
    # =====================================================================
    d_stage_quality = 0.0
    # Better circadian alignment → more deep + REM sleep
    d_stage_quality += (circadian - 0.6) * 4.0  # up to +1.6 for optimal; −~1.5 for extremely late
    # Good sleep duration boosts stage quality
    if hours >= 7.0 and hours <= 9.0:
        d_stage_quality += 0.8 * circadian_sleep_bonus
    elif hours < 6.0:
        d_stage_quality -= 1.5
    # Flexibility significantly improves deep / REM; intense late exercise suppresses them
    if exercise_type == ExerciseType.FLEXIBILITY:
        d_stage_quality += 1.0 * dur_factor
    elif is_intense and circadian < 0.6:
        d_stage_quality -= 1.0  # high intensity + late bedtime → reduced deep sleep
    if current.stress_avg > 60:
        d_stage_quality -= 0.8
    d_stage_quality += rng.gauss(0, 2.5)

    # =====================================================================
    # VO2max (slow-moving, responds to weeks of aerobic training)
    # =====================================================================
    d_vo2 = 0.0
    circadian_multiplier = getattr(rm, "circadian_sensitivity", 1.0)
    # Only cardio and HIIT with sufficient duration move VO2max meaningfully
    if exercise_type in {ExerciseType.CARDIO, ExerciseType.HIIT}:
        minutes = EXERCISE_DURATION_MINUTES[exercise_duration]
        if minutes >= 30:
            base_vo2_gain = 0.10 if exercise_type == ExerciseType.CARDIO else 0.18
            d_vo2 += base_vo2_gain * (dur_factor / 1.5) * debt_factor
    if is_overtraining:
        d_vo2 -= 0.05 * (1.0 + overtraining_amp * 0.2)
    # Noise is very small (VO2max doesn't change dramatically day to day)
    d_vo2 += rng.gauss(0, 0.05)

    return BiomarkerDeltas(
        resting_hr=round(d_rhr, 3),
        hrv=round(d_hrv, 3),
        sleep_score=round(d_sleep, 3),
        stress_avg=round(d_stress, 3),
        body_battery=round(d_battery, 3),
        sleep_stage_quality=round(d_stage_quality, 3),
        vo2_max=round(d_vo2, 4),
    )


def apply_deltas(current: Biomarkers, deltas: BiomarkerDeltas) -> Biomarkers:
    """Apply biomarker deltas to current state, clamping to valid ranges."""
    return Biomarkers(
        resting_hr=round(_clamp(current.resting_hr + deltas.resting_hr, 40.0, 120.0), 2),
        hrv=round(_clamp(current.hrv + deltas.hrv, 5.0, 150.0), 2),
        sleep_score=round(_clamp(current.sleep_score + deltas.sleep_score, 0.0, 100.0), 2),
        stress_avg=round(_clamp(current.stress_avg + deltas.stress_avg, 0.0, 100.0), 2),
        body_battery=round(_clamp(current.body_battery + deltas.body_battery, 0.0, 100.0), 2),
        sleep_stage_quality=round(
            _clamp(current.sleep_stage_quality + deltas.sleep_stage_quality, 0.0, 100.0), 2
        ),
        vo2_max=round(_clamp(current.vo2_max + deltas.vo2_max, 10.0, 90.0), 3),
    )


def apply_life_event(
    action: Action, rng: stdlib_random.Random
) -> Action:
    """5% chance of a random life disruption per day."""
    if rng.random() > 0.05:
        return action

    event = rng.choice(["bad_sleep", "missed_activity"])
    if event == "bad_sleep":
        return Action(
            sleep=SleepDuration.VERY_SHORT,
            activity=action.activity,
            bedtime=action.bedtime,
            exercise_type=action.exercise_type,
            exercise_duration=action.exercise_duration,
        )
    else:  # missed activity
        return Action(
            sleep=action.sleep,
            activity=ActivityLevel.REST_DAY,
            bedtime=action.bedtime,
            exercise_type=ExerciseType.NONE,
            exercise_duration=ExerciseDuration.NONE,
        )
