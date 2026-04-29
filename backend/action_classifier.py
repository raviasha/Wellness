"""Classify Garmin wearable data into discrete sleep/activity actions.

Maps raw Garmin sync data (sleep_duration_hours, active_minutes, exercise_type,
exercise_duration_minutes, sleep_start_hour, etc.) into the 5-dimensional
MultiDiscrete([5,5,5,5,5]) action space used by the wellness environment.
"""
from __future__ import annotations

from wellness_env.models import (
    Action, ActivityLevel, SleepDuration,
    BedtimeWindow, ExerciseDuration, ExerciseType,
    GARMIN_ACTIVITY_TYPE_MAP,
)


def classify_sleep(sleep_duration_hours: float | None) -> SleepDuration:
    """Map sleep duration in hours to a SleepDuration enum."""
    if sleep_duration_hours is None:
        return SleepDuration.OPTIMAL_LOW
    h = sleep_duration_hours
    if h < 6.0:
        return SleepDuration.VERY_SHORT
    if h < 7.0:
        return SleepDuration.SHORT
    if h < 8.0:
        return SleepDuration.OPTIMAL_LOW
    if h < 9.0:
        return SleepDuration.OPTIMAL_HIGH
    return SleepDuration.LONG


def classify_bedtime(sleep_start_hour: float | None) -> BedtimeWindow:
    """Map decimal bedtime hour (e.g. 22.5 = 10:30 PM) to a BedtimeWindow enum.

    Windows (local time):
        EARLY         : before 21:00
        OPTIMAL       : 21:00 – 22:59
        LATE          : 23:00 – 23:59
        VERY_LATE     : 00:00 – 01:29
        EXTREMELY_LATE: 01:30+
    Hours are normalised modulo 24, so values like 0.5 (12:30 AM) work correctly.
    """
    if sleep_start_hour is None:
        return BedtimeWindow.OPTIMAL
    h = sleep_start_hour % 24
    # Treat early-morning hours (< 6) as past-midnight late bedtimes
    if h < 1.5 or (h >= 24):
        return BedtimeWindow.VERY_LATE
    if h < 6.0:
        return BedtimeWindow.EXTREMELY_LATE
    if h < 21.0:
        return BedtimeWindow.EARLY
    if h < 23.0:
        return BedtimeWindow.OPTIMAL
    if h < 24.0:
        return BedtimeWindow.LATE
    return BedtimeWindow.OPTIMAL


def classify_activity(
    active_minutes: float | None = None,
    active_calories: float | None = None,
    steps: int | None = None,
) -> ActivityLevel:
    """Map Garmin activity metrics to an ActivityLevel enum."""
    mins = active_minutes or 0.0
    cals = active_calories or 0.0
    st = steps or 0

    if mins >= 60:
        return ActivityLevel.HIGH_INTENSITY
    if mins >= 30:
        return ActivityLevel.VIGOROUS_ACTIVITY
    if mins >= 15:
        return ActivityLevel.MODERATE_ACTIVITY
    if mins >= 5:
        return ActivityLevel.LIGHT_ACTIVITY

    if cals >= 500:
        return ActivityLevel.VIGOROUS_ACTIVITY
    if cals >= 300:
        return ActivityLevel.MODERATE_ACTIVITY
    if cals >= 100:
        return ActivityLevel.LIGHT_ACTIVITY

    if st >= 10000:
        return ActivityLevel.MODERATE_ACTIVITY
    if st >= 5000:
        return ActivityLevel.LIGHT_ACTIVITY

    return ActivityLevel.REST_DAY


def classify_exercise_type(exercise_type_key: str | None) -> ExerciseType:
    """Map a Garmin activityType.typeKey string to an ExerciseType enum.

    Falls back to ExerciseType.NONE when the key is absent or unrecognised.
    """
    if not exercise_type_key:
        return ExerciseType.NONE
    return GARMIN_ACTIVITY_TYPE_MAP.get(exercise_type_key.lower(), ExerciseType.NONE)


def classify_exercise_duration(exercise_duration_minutes: int | float | None) -> ExerciseDuration:
    """Map exercise duration in minutes to an ExerciseDuration enum.

    Thresholds:
        NONE     :  < 5 min
        SHORT    :  5 – 19 min
        MODERATE : 20 – 44 min
        LONG     : 45 – 74 min
        EXTENDED : 75+ min
    """
    if exercise_duration_minutes is None:
        return ExerciseDuration.NONE
    mins = float(exercise_duration_minutes)
    if mins < 5:
        return ExerciseDuration.NONE
    if mins < 20:
        return ExerciseDuration.SHORT
    if mins < 45:
        return ExerciseDuration.MODERATE
    if mins < 75:
        return ExerciseDuration.LONG
    return ExerciseDuration.EXTENDED


def classify_daily_actions(sync_row: dict) -> Action:
    """Classify a full day's Garmin sync data into a 5-dimensional Action.

    Args:
        sync_row: dict with optional keys:
            sleep_duration_hours, sleep_start_hour,
            active_minutes, active_calories, steps,
            exercise_type (Garmin typeKey string),
            exercise_duration_minutes.

    Returns:
        Action with all 5 fields classified.
    """
    sleep = classify_sleep(sync_row.get("sleep_duration_hours"))
    bedtime = classify_bedtime(sync_row.get("sleep_start_hour"))
    activity = classify_activity(
        active_minutes=sync_row.get("active_minutes"),
        active_calories=sync_row.get("active_calories"),
        steps=sync_row.get("steps"),
    )
    exercise_type = classify_exercise_type(sync_row.get("exercise_type"))
    exercise_duration = classify_exercise_duration(sync_row.get("exercise_duration_minutes"))

    return Action(
        sleep=sleep,
        bedtime=bedtime,
        activity=activity,
        exercise_type=exercise_type,
        exercise_duration=exercise_duration,
    )
