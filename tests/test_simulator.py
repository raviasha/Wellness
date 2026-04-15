"""Tests for the physiological simulator (biomarker transition dynamics)."""
from __future__ import annotations

import random

import pytest

from wellness_env.models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    ExerciseType,
    NutritionType,
    SleepDuration,
    SLEEP_HOURS,
)
from wellness_env.personas import PERSONAS
from wellness_env.simulator import (
    apply_deltas,
    apply_life_event,
    compute_biomarker_changes,
)


# ============================================================================
# Helpers
# ============================================================================

SEED = 42

BASELINE_BIOMARKERS = Biomarkers(
    resting_hr=70, hrv=40, vo2_max=30, body_fat_pct=25,
    lean_mass_kg=55, sleep_efficiency=75, cortisol_proxy=50, energy_level=50,
)

GOOD_ACTION = Action(
    sleep=SleepDuration.OPTIMAL_LOW,
    exercise=ExerciseType.MODERATE_CARDIO,
    nutrition=NutritionType.BALANCED,
)

BAD_ACTION = Action(
    sleep=SleepDuration.VERY_SHORT,
    exercise=ExerciseType.NONE,
    nutrition=NutritionType.PROCESSED,
)

YOGA_ACTION = Action(
    sleep=SleepDuration.OPTIMAL_HIGH,
    exercise=ExerciseType.YOGA,
    nutrition=NutritionType.BALANCED,
)


# ============================================================================
# compute_biomarker_changes
# ============================================================================

class TestComputeBiomarkerChanges:
    def test_returns_biomarker_deltas(self):
        rng = random.Random(SEED)
        persona = PERSONAS["athletic_performance"]
        deltas = compute_biomarker_changes(
            GOOD_ACTION, BASELINE_BIOMARKERS, persona, [], rng,
        )
        assert isinstance(deltas, BiomarkerDeltas)

    def test_all_eight_fields_present(self):
        rng = random.Random(SEED)
        persona = PERSONAS["athletic_performance"]
        deltas = compute_biomarker_changes(
            GOOD_ACTION, BASELINE_BIOMARKERS, persona, [], rng,
        )
        d = deltas.model_dump()
        expected = {
            "resting_hr", "hrv", "vo2_max", "body_fat_pct",
            "lean_mass_kg", "sleep_efficiency", "cortisol_proxy", "energy_level",
        }
        assert set(d.keys()) == expected

    def test_good_sleep_improves_hrv(self):
        """Sleeping 7-8h should generally improve HRV."""
        rng = random.Random(SEED)
        persona = PERSONAS["stress_management"]  # high HRV sleep sensitivity
        action = Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.NONE,
            nutrition=NutritionType.BALANCED,
        )
        # Run multiple times and average to overcome noise
        total_hrv = 0.0
        runs = 50
        for i in range(runs):
            rng = random.Random(SEED + i)
            d = compute_biomarker_changes(action, BASELINE_BIOMARKERS, persona, [], rng)
            total_hrv += d.hrv
        avg_hrv = total_hrv / runs
        assert avg_hrv > 0, f"Average HRV delta {avg_hrv} should be positive with good sleep"

    def test_cardio_improves_vo2(self):
        """Cardio exercise should generally improve VO2 max."""
        persona = PERSONAS["athletic_performance"]
        total_vo2 = 0.0
        runs = 50
        for i in range(runs):
            rng = random.Random(SEED + i)
            d = compute_biomarker_changes(GOOD_ACTION, BASELINE_BIOMARKERS, persona, [], rng)
            total_vo2 += d.vo2_max
        avg_vo2 = total_vo2 / runs
        assert avg_vo2 > 0, f"Average VO2 delta {avg_vo2} should be positive with cardio"

    def test_yoga_reduces_cortisol(self):
        """Yoga should reduce cortisol on average."""
        persona = PERSONAS["stress_management"]
        total_cortisol = 0.0
        runs = 50
        for i in range(runs):
            rng = random.Random(SEED + i)
            d = compute_biomarker_changes(YOGA_ACTION, BASELINE_BIOMARKERS, persona, [], rng)
            total_cortisol += d.cortisol_proxy
        avg_cortisol = total_cortisol / runs
        assert avg_cortisol < 0, f"Average cortisol delta {avg_cortisol} should be negative with yoga"

    def test_different_personas_produce_different_deltas(self):
        """Same action should produce different results for different personas."""
        rng1 = random.Random(SEED)
        rng2 = random.Random(SEED)
        d1 = compute_biomarker_changes(
            GOOD_ACTION, BASELINE_BIOMARKERS, PERSONAS["athletic_performance"], [], rng1,
        )
        d2 = compute_biomarker_changes(
            GOOD_ACTION, BASELINE_BIOMARKERS, PERSONAS["weight_loss"], [], rng2,
        )
        # At least some biomarker deltas should differ due to different ResponseModels
        d1_dict = d1.model_dump()
        d2_dict = d2.model_dump()
        diffs = [abs(d1_dict[k] - d2_dict[k]) for k in d1_dict]
        assert any(d > 0.001 for d in diffs), "Personas should produce different deltas"

    def test_overtraining_effect(self):
        """Consecutive intense exercise days should trigger overtraining effects."""
        persona = PERSONAS["weight_loss"]  # threshold=2
        # Build history with 2 consecutive intense days
        history = [
            {"actual_action": {"exercise": "hiit", "sleep": "7_to_8h"}},
            {"actual_action": {"exercise": "hiit", "sleep": "7_to_8h"}},
        ]
        hiit_action = Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.HIIT,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
        # Overtraining should spike cortisol
        total_cortisol = 0.0
        runs = 50
        for i in range(runs):
            rng = random.Random(SEED + i)
            d = compute_biomarker_changes(hiit_action, BASELINE_BIOMARKERS, persona, history, rng)
            total_cortisol += d.cortisol_proxy
        avg_cortisol = total_cortisol / runs
        assert avg_cortisol > 0, f"Overtraining should raise cortisol, got {avg_cortisol}"


# ============================================================================
# apply_deltas
# ============================================================================

class TestApplyDeltas:
    def test_applies_deltas_correctly(self):
        deltas = BiomarkerDeltas(
            resting_hr=-1.0, hrv=2.0, vo2_max=0.5, body_fat_pct=-0.1,
            lean_mass_kg=0.05, sleep_efficiency=1.0, cortisol_proxy=-3.0, energy_level=5.0,
        )
        result = apply_deltas(BASELINE_BIOMARKERS, deltas)
        assert result.resting_hr == 69.0
        assert result.hrv == 42.0
        assert result.vo2_max == 30.5
        assert abs(result.body_fat_pct - 24.9) < 0.01
        assert abs(result.lean_mass_kg - 55.05) < 0.01
        assert result.sleep_efficiency == 76.0
        assert result.cortisol_proxy == 47.0
        assert result.energy_level == 55.0

    def test_clamps_to_valid_ranges(self):
        extreme_deltas = BiomarkerDeltas(
            resting_hr=-999, hrv=999, vo2_max=999, body_fat_pct=-999,
            lean_mass_kg=999, sleep_efficiency=999, cortisol_proxy=-999, energy_level=999,
        )
        result = apply_deltas(BASELINE_BIOMARKERS, extreme_deltas)
        assert result.resting_hr == 40.0  # clamped to min
        assert result.hrv == 150.0        # clamped to max
        assert result.vo2_max == 70.0
        assert result.body_fat_pct == 3.0
        assert result.lean_mass_kg == 100.0
        assert result.sleep_efficiency == 100.0
        assert result.cortisol_proxy == 0.0
        assert result.energy_level == 100.0

    def test_negative_clamp(self):
        extreme_deltas = BiomarkerDeltas(
            resting_hr=999, hrv=-999, vo2_max=-999, body_fat_pct=999,
            lean_mass_kg=-999, sleep_efficiency=-999, cortisol_proxy=999, energy_level=-999,
        )
        result = apply_deltas(BASELINE_BIOMARKERS, extreme_deltas)
        assert result.resting_hr == 120.0
        assert result.hrv == 5.0
        assert result.vo2_max == 15.0
        assert result.body_fat_pct == 50.0
        assert result.lean_mass_kg == 30.0
        assert result.sleep_efficiency == 0.0
        assert result.cortisol_proxy == 100.0
        assert result.energy_level == 0.0


# ============================================================================
# apply_life_event
# ============================================================================

class TestApplyLifeEvent:
    def test_usually_no_change(self):
        """95% of the time, action should be unchanged."""
        unchanged = 0
        total = 1000
        for i in range(total):
            rng = random.Random(SEED + i)
            result = apply_life_event(GOOD_ACTION, rng)
            if result == GOOD_ACTION:
                unchanged += 1
        # Should be roughly 950 ± ~30
        assert unchanged > 900, f"Only {unchanged}/1000 unchanged (expected ~950)"
        assert unchanged < 990, f"{unchanged}/1000 unchanged — life events never trigger?"

    def test_life_event_changes_action(self):
        """When a life event triggers, at least one action field changes."""
        changed = False
        for i in range(1000):
            rng = random.Random(SEED + i)
            result = apply_life_event(GOOD_ACTION, rng)
            if result != GOOD_ACTION:
                changed = True
                # At least one field must differ
                assert (
                    result.sleep != GOOD_ACTION.sleep
                    or result.exercise != GOOD_ACTION.exercise
                    or result.nutrition != GOOD_ACTION.nutrition
                )
                break
        assert changed, "No life event triggered in 1000 attempts"


# ============================================================================
# SLEEP_HOURS mapping
# ============================================================================

class TestSleepHours:
    def test_all_durations_mapped(self):
        for sd in SleepDuration:
            assert sd in SLEEP_HOURS

    def test_hours_are_ordered(self):
        hours = [SLEEP_HOURS[sd] for sd in [
            SleepDuration.VERY_SHORT,
            SleepDuration.SHORT,
            SleepDuration.OPTIMAL_LOW,
            SleepDuration.OPTIMAL_HIGH,
            SleepDuration.LONG,
        ]]
        for i in range(len(hours) - 1):
            assert hours[i] < hours[i + 1], f"Hours not ordered: {hours}"
