# RL Wellness App — Round 1 Requirements (OpenEnv Hackathon)

This document is a **focused slice** of the full vision ([requirements-round2.md](requirements-round2.md)) scoped to the Scaler/Meta PyTorch Hackathon Round 1 deliverables: an OpenEnv-compliant wellness environment deployed to Hugging Face Spaces.

---

## 1. Scope & Goal

Build an **Outcome-Based Wellness Simulator Environment** where an RL/LLM agent guides a simulated user toward better health outcomes — measured by **biomarker changes** (resting HR, HRV, VO2 max, body fat %, lean mass, sleep efficiency, cortisol proxy, energy level) — over a multi-day episode. The agent takes actions (sleep, exercise, nutrition recommendations) and observes **outcome deltas**, learning what action sequences produce the best biomarker improvements for each persona. The environment exposes the OpenEnv interface (`step()`, `reset()`, `state()`) with Pydantic-typed models.

**Key design principle:** There are no fixed action-category scores. The reward is computed entirely from goal-weighted biomarker deltas. Each persona has a **hidden physiological response model** that the agent must learn through experience — the same action produces different outcomes for different people.

**What Round 1 delivers:**
- OpenEnv-compliant environment (the simulator + outcome-based payoff function)
- 3 tasks with programmatic graders (easy → medium → hard)
- Baseline `inference.py` using an LLM agent via OpenAI API
- Dockerfile for Hugging Face Spaces deployment
- `openenv.yaml` metadata

**What Round 1 does NOT include** (deferred to Round 2+):
- User-facing app / LLM feature extractor (§5 of full doc)
- Multi-agent training visualization (§6 of full doc)
- Trained NN policy (DQN/PPO/SAC) — Round 1 agent is an LLM
- Production history store, observability, privacy (§11–13 of full doc)

---

## 2. OpenEnv Interface

The environment must implement the OpenEnv specification. All inputs and outputs are Pydantic models.

### 2.1 Core Methods

```python
class WellnessEnv:
    def reset(self, task_name: str) -> Observation:
        """Initialize a new episode for the given task. Returns initial observation."""

    def step(self, action: Action) -> StepResult:
        """Apply action, advance one day, return (observation, reward, done, info)."""

    def state(self) -> EnvState:
        """Return the full internal state (for debugging/grading). Not seen by agent."""
```

### 2.2 Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# --- Enums ---

class SleepDuration(str, Enum):
    VERY_SHORT = "less_than_6h"
    SHORT = "6_to_7h"
    OPTIMAL_LOW = "7_to_8h"
    OPTIMAL_HIGH = "8_to_9h"
    LONG = "more_than_9h"

class ExerciseType(str, Enum):
    NONE = "none"
    LIGHT_CARDIO = "light_cardio"
    MODERATE_CARDIO = "moderate_cardio"
    HIIT = "hiit"
    STRENGTH = "strength"
    YOGA = "yoga"

class NutritionType(str, Enum):
    HIGH_PROTEIN = "high_protein"
    BALANCED = "balanced"
    HIGH_CARB = "high_carb"
    PROCESSED = "processed"
    SKIPPED = "skipped"

class Goal(str, Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    OVERALL_WELLNESS = "overall_wellness"
    LONGEVITY = "longevity"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    STRESS_MANAGEMENT = "stress_management"

# --- Action ---

class Action(BaseModel):
    sleep: SleepDuration = Field(description="Target sleep duration category")
    exercise: ExerciseType = Field(description="Exercise type for today")
    nutrition: NutritionType = Field(description="Nutrition category for today")

# --- Biomarkers (the outcome variables) ---

class Biomarkers(BaseModel):
    """Measurable health outcomes."""
    resting_hr: float = Field(ge=40, le=120, description="Resting heart rate (bpm)")
    hrv: float = Field(ge=5, le=150, description="Heart rate variability (ms RMSSD)")
    vo2_max: float = Field(ge=15, le=70, description="Estimated VO2 max (ml/kg/min)")
    body_fat_pct: float = Field(ge=3, le=50, description="Body fat percentage")
    lean_mass_kg: float = Field(ge=30, le=100, description="Lean body mass (kg)")
    sleep_efficiency: float = Field(ge=0, le=100, description="Sleep efficiency %")
    cortisol_proxy: float = Field(ge=0, le=100, description="Stress proxy (0=low, 100=high)")
    energy_level: float = Field(ge=0, le=100, description="Subjective energy (0-100)")

class BiomarkerDeltas(BaseModel):
    """Change in each biomarker since last step. This IS the outcome signal."""
    resting_hr: float = Field(description="Δ resting HR (negative = improving)")
    hrv: float = Field(description="Δ HRV (positive = improving)")
    vo2_max: float = Field(description="Δ VO2 max (positive = improving)")
    body_fat_pct: float = Field(description="Δ body fat % (negative = improving for weight loss)")
    lean_mass_kg: float = Field(description="Δ lean mass (positive = improving for muscle gain)")
    sleep_efficiency: float = Field(description="Δ sleep efficiency (positive = improving)")
    cortisol_proxy: float = Field(description="Δ cortisol proxy (negative = improving)")
    energy_level: float = Field(description="Δ energy (positive = improving)")

# --- Observation (what the agent sees) ---

class OutcomeTrends(BaseModel):
    """7-day trends for outcome variables."""
    resting_hr_trend: float = Field(description="7-day slope of resting HR")
    hrv_trend: float = Field(description="7-day slope of HRV")
    vo2_max_trend: float = Field(description="7-day slope of VO2 max")
    body_fat_trend: float = Field(description="7-day slope of body fat %")
    lean_mass_trend: float = Field(description="7-day slope of lean mass")
    sleep_efficiency_trend: float = Field(description="7-day slope of sleep efficiency")
    reward_trend: float = Field(description="7-day slope of total reward")
    reward_consistency: float = Field(description="Stddev of reward over last 7 days")

class Observation(BaseModel):
    day: int = Field(ge=0, description="Current day in the episode")
    total_days: int = Field(description="Total days in this episode")
    goal: Goal = Field(description="User's primary health goal")
    biomarkers: Biomarkers = Field(description="Current biomarker values")
    deltas: BiomarkerDeltas = Field(description="Change since last step")
    trends: Optional[OutcomeTrends] = Field(default=None, description="7-day outcome trends (available after day 7)")
    persona_name: str = Field(description="Name of the simulated user persona")
    compliance_rate: float = Field(ge=0, le=1, description="How often this persona follows recommendations")

# --- Reward (outcome-based) ---

class RewardBreakdown(BaseModel):
    resting_hr_reward: float = Field(description="Reward from resting HR change")
    hrv_reward: float = Field(description="Reward from HRV change")
    vo2_max_reward: float = Field(description="Reward from VO2 max change")
    body_fat_reward: float = Field(description="Reward from body fat change")
    lean_mass_reward: float = Field(description="Reward from lean mass change")
    sleep_efficiency_reward: float = Field(description="Reward from sleep efficiency change")
    cortisol_reward: float = Field(description="Reward from cortisol proxy change")
    energy_reward: float = Field(description="Reward from energy change")
    total: float = Field(description="Goal-weighted total reward (0-100)")

# --- Step Result ---

class StepResult(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool = Field(description="Whether the episode has ended")
    info: dict = Field(default_factory=dict, description="Additional metadata (actual vs recommended action, compliance, etc.)")

# --- Full Environment State (for grading) ---

class EnvState(BaseModel):
    day: int
    total_days: int
    goal: Goal
    persona_name: str
    compliance_rate: float
    biomarkers: Biomarkers
    history: list  # Full day-by-day history
    cumulative_reward: float
    action_space_description: str = Field(default="sleep × exercise × nutrition = 5 × 6 × 5 = 150 combos")
```

### 2.3 `openenv.yaml`

```yaml
tasks:
  - name: single_goal
    description: "Easy: Optimize performance biomarkers over 14 days"
    difficulty: easy

  - name: multi_outcome
    description: "Medium: Balance all 8 biomarkers for stress management over 30 days"
    difficulty: medium

  - name: resistant_adaptation
    description: "Hard: Improve outcomes for a low-compliance weight-loss persona over 30 days"
    difficulty: hard

environment:
  name: wellness-outcome
  description: >
    Outcome-based wellness simulator where rewards are driven by measured
    biomarker changes (resting HR, HRV, VO2 max, body fat %, lean mass,
    sleep efficiency, cortisol proxy, energy level) weighted by the user's
    health goal. Each persona has hidden physiological response parameters
    the agent must learn through experience.

entrypoint: python inference.py
```

---

## 3. Biomarker Outcomes — The Reward Signal

For Round 1, the reward is **entirely outcome-based**. There are no fixed action-category scores. Instead, the agent observes 8 biomarkers and is rewarded based on how its actions change those biomarkers, weighted by the user's health goal.

### 3.1 Biomarker Definitions

| Biomarker | Direction | Range | What It Measures |
|-----------|-----------|-------|------------------|
| `resting_hr` | Lower is better | 40–120 bpm | Cardiovascular fitness |
| `hrv` | Higher is better | 5–150 ms | Autonomic nervous system health |
| `vo2_max` | Higher is better | 15–70 ml/kg/min | Aerobic capacity |
| `body_fat_pct` | Lower is better* | 3–50% | Body composition |
| `lean_mass_kg` | Higher is better* | 30–100 kg | Muscle mass |
| `sleep_efficiency` | Higher is better | 0–100% | Sleep quality |
| `cortisol_proxy` | Lower is better | 0–100 | Stress level |
| `energy_level` | Higher is better | 0–100 | Subjective energy |

*Importance varies by goal (e.g., body fat is weighted 0.35 for weight loss but 0.05 for athletic performance).

### 3.2 Goal-Specific Weighting

Each goal defines different weights for each biomarker. All weights for a goal sum to 1.0:

| Biomarker | Weight Loss | Muscle Gain | Overall Wellness | Longevity | Athletic Perf. | Stress Mgmt |
|-----------|-------------|-------------|------------------|-----------|----------------|-------------|
| `resting_hr` | 0.05 | 0.05 | 0.125 | 0.15 | 0.10 | 0.10 |
| `hrv` | 0.05 | 0.05 | 0.125 | 0.20 | 0.10 | 0.20 |
| `vo2_max` | 0.05 | 0.10 | 0.125 | 0.25 | 0.30 | 0.05 |
| `body_fat_pct` | 0.35 | 0.10 | 0.125 | 0.10 | 0.05 | 0.05 |
| `lean_mass_kg` | 0.10 | 0.35 | 0.125 | 0.05 | 0.20 | 0.05 |
| `sleep_efficiency` | 0.10 | 0.10 | 0.125 | 0.10 | 0.05 | 0.20 |
| `cortisol_proxy` | 0.10 | 0.05 | 0.125 | 0.10 | 0.05 | 0.25 |
| `energy_level` | 0.20 | 0.20 | 0.125 | 0.05 | 0.15 | 0.10 |

### 3.3 Actions (the Agent's Levers)

Sleep, exercise, and nutrition exist only as the **action space** — they are levers the agent can pull, not reward categories:

| Action Dimension | Options | Count |
|-----------------|---------|-------|
| Sleep | `<6h`, `6-7h`, `7-8h`, `8-9h`, `>9h` | 5 |
| Exercise | `none`, `light_cardio`, `moderate_cardio`, `hiit`, `strength`, `yoga` | 6 |
| Nutrition | `high_protein`, `balanced`, `high_carb`, `processed`, `skipped` | 5 |

Total: 5 × 6 × 5 = **150 action combinations** per time step.

### 3.4 Cross-Action Interactions (Hidden in Simulator)

These interactions are encoded in the simulator's biomarker transition model. They are **not visible to the agent** — the agent must discover them through observed outcomes:

| Interaction | Biomarker Effect |
|---|---|
| Intense exercise + sleep debt | Exercise gains reduced (debt_factor multiplier) |
| Overtraining (3+ consecutive intense days) | Cortisol spike, HRV depression, energy drain |
| High protein + intense exercise | Enhanced recovery (protein_recovery_multiplier) |
| Poor sleep (<6h) | Resting HR increases, HRV drops, cortisol rises, fat loss impaired |
| Yoga | Cortisol reduction, HRV boost, mild sleep improvement |
| High cortisol (>60) | HRV depressed, sleep quality drops, energy drains |

---

## 4. Payoff Function (Outcome-Based Reward)

### 4.1 Per-Step Reward

$$R_t = 50 + \sum_{i=1}^{8} \text{normalize}(\Delta_i) \times w_{i,\text{goal}} \times 100$$

Clamped to $[0, 100]$. Baseline (no biomarker change) = 50. Improvements push above 50, declines push below 50.

### 4.2 Reward Computation

For each biomarker:

1. **Raw delta**: The change in the biomarker from the previous step (computed by the simulator)
2. **Sign normalization**: For "lower is better" markers (`resting_hr`, `body_fat_pct`, `cortisol_proxy`), the delta is negated so positive always = improving
3. **Scale normalization**: Divide by a `DELTA_SCALE` that defines what an "excellent" daily improvement looks like
4. **Clamp**: Normalized value clamped to [-2, +2] to prevent outlier noise from dominating
5. **Weight**: Multiply by the goal-specific weight

**Delta scales (what constitutes "excellent" daily improvement):**

| Biomarker | Delta Scale | Meaning |
|-----------|------------|---------|
| `resting_hr` | 1.0 bpm | −1 bpm/day is excellent |
| `hrv` | 5.0 ms | +5 ms/day is excellent |
| `vo2_max` | 0.3 ml/kg/min | +0.3/day is excellent |
| `body_fat_pct` | 0.05% | −0.05%/day is excellent |
| `lean_mass_kg` | 0.1 kg | +0.1 kg/day is excellent |
| `sleep_efficiency` | 2.0% | +2%/day is excellent |
| `cortisol_proxy` | 5.0 pts | −5 pts/day is excellent |
| `energy_level` | 10.0 pts | +10 pts/day is excellent |

### 4.3 Example: Weight Loss Goal

For a weight loss user whose body fat dropped 0.04% and energy rose 8 points:

- `body_fat_pct` delta = -0.04% → normalized = +0.04/0.05 = +0.8 → weighted = 0.8 × 0.35 × 100 = +28.0
- `energy_level` delta = +8 → normalized = 8/10 = +0.8 → weighted = 0.8 × 0.20 × 100 = +16.0
- Other markers contribute smaller amounts
- Total ≈ 50 + 28 + 16 + (others) → reward well above 50

### 4.4 Key Differences from Action-Category Scoring

| Aspect | Action-Category (old design) | Outcome-Based (current) |
|--------|--------------------------|------------------------|
| **Reward signal** | Fixed: sleep_score, exercise_score, nutrition_score → weighted average | Dynamic: computed from measured biomarker **deltas** |
| **What's scored** | The action itself ("was this a good exercise choice?") | The outcome ("did biomarkers improve?") |
| **Personalization** | Same score for same action regardless of person | Same action → different outcomes per persona → different rewards |
| **Agent learning** | Learns "good actions" directly | Learns "what action sequences lead to good outcomes" via experience |

---

## 5. Simulator (Biomarker Transition Dynamics)

The simulator models how today's actions change each of the 8 biomarkers. It is the **hidden function the agent must learn** through observed outcome deltas. Each persona has a unique `ResponseModel` that defines their physiological response — the agent never sees these parameters.

### 5.1 State Transition

```
step(state, action):
    1. Apply persona compliance filter (may override action)
    2. Apply life event noise (5% chance of disruption)
    3. Compute biomarker deltas using the persona's hidden ResponseModel
    4. Apply deltas to get new biomarker state (clamped to valid ranges)
    5. Compute outcome-based reward from deltas (goal-weighted)
    6. Append to history
    7. Compute 7-day outcome trends (if day >= 7)
    8. Check if done (day >= total_days)
    9. Return (observation, reward, done, info)
```

### 5.2 Biomarker Response Model

Each biomarker is influenced by a combination of actions with persona-specific sensitivity parameters:

| Biomarker | Main Action Drivers | Cross-Effects |
|-----------|-------------------|---------------|
| **Resting HR** | Sleep ≥7h improves; exercise (non-overtraining) improves | Overtraining raises it; poor sleep raises it |
| **HRV** | Sleep duration; yoga (+2ms); moderate sleep sensitivity | Overtraining tanks it; high cortisol depresses it |
| **VO2 Max** | Cardio exercise (HIIT > moderate > light); slow decay without exercise | Sleep debt reduces exercise efficacy |
| **Body Fat %** | Exercise intensity; nutrition quality | Poor sleep impairs fat loss; protein accelerates loss |
| **Lean Mass** | Strength training + protein; HIIT at 50% strength effect | Inactivity + poor nutrition → muscle loss |
| **Sleep Efficiency** | Optimal sleep hours (7-9h); moderate cardio/yoga improve | Intense exercise hurts; high cortisol hurts |
| **Cortisol** | Sleep recovery; yoga (-3); overtraining spike | Poor nutrition raises; natural regression if >50 |
| **Energy** | Sleep quality; nutrition quality; light exercise boosts | Overtraining drains; sleep debt drains; high cortisol drains |

### 5.3 Hidden ResponseModel Parameters

Each persona has ~15 parameters that control their unique response. These are the parameters the agent **cannot see** and must learn from experience:

```python
@dataclass
class ResponseModel:
    # Sleep response
    hrv_sleep_sensitivity: float        # HRV improvement per hour above 7h
    rhr_sleep_benefit: float            # RHR drop per optimal sleep night
    sleep_efficiency_base: float        # Baseline sleep efficiency
    cortisol_sleep_recovery: float      # Cortisol reduction per optimal sleep night

    # Exercise response
    vo2_cardio_gain: float              # VO2 improvement per cardio session
    lean_mass_strength_gain: float      # Lean mass gain per strength session
    body_fat_exercise_loss: float       # Body fat loss per intense session
    rhr_exercise_benefit: float         # RHR improvement per exercise session
    cortisol_exercise_stress: float     # Cortisol rise from intense exercise
    overtraining_threshold: int         # Consecutive intense days before harm

    # Nutrition response
    body_fat_nutrition_sensitivity: float
    lean_mass_protein_gain: float
    energy_nutrition_sensitivity: float
    cortisol_nutrition_stress: float

    # Cross-action
    sleep_debt_exercise_penalty: float  # Exercise gain reduction per debt hour
    protein_recovery_multiplier: float  # Protein boost to post-exercise recovery
    overtraining_cortisol_spike: float
```

### 5.4 Persona Compliance Model

When the agent recommends an action, the simulated user follows it with probability $p_{comply}$:

```
if random() < p_comply:
    actual_action = recommended_action
elif random() < 0.6:
    actual_action = persona_default_action    # revert to old habits
else:
    actual_action = random_action()           # something unrelated
```

The `info` dict in `StepResult` reports `recommended_action` vs `actual_action` so the agent can observe compliance.

### 5.5 Stochasticity

- Action noise: actual sleep duration sampled ±0.5h around category midpoint
- Gaussian noise: each biomarker delta gets random perturbation (e.g., ±0.3 for RHR, ±1.5 for HRV, ±0.05 for VO2 max)
- Life events: 5% chance per day of a random disruption (bad sleep, missed exercise, social dinner)

---

## 6. Tasks & Graders

### 6.1 Task Design Principles

- Each task is a **configuration** of the same environment (persona, goal, duration)
- Graders score 0.0–1.0, are **programmatic and deterministic** (same actions → same score)
- Graders evaluate **biomarker outcomes**, not action quality — did the biomarkers actually improve?
- Difficulty scales by: episode length, persona compliance rate, response model complexity

### 6.2 Task 1: Single-Goal Focus (Easy) — `single_goal`

**Configuration:**
| Parameter | Value |
|---|---|
| Persona | `athletic_performance` |
| Goal | `ATHLETIC_PERFORMANCE` |
| Compliance rate | 0.7 |
| Episode length | 14 days |

**Grader:**
```
primary_biomarker = highest-weighted biomarker for the goal (e.g., vo2_max for ATHLETIC_PERFORMANCE)
Score = 0.6 * normalize(avg_reward, 40, 75)
     + 0.2 * normalize(primary_biomarker_improvement, 0, scale)
     + 0.2 * normalize(reward_trend, -0.5, 1.0)
```
- 60% weight on average reward (reflects overall biomarker improvement)
- 20% weight on primary biomarker improvement (the goal-specific metric)
- 20% weight on positive trend (is the agent learning to improve over the episode?)
- Expected scores: Random ~0.2–0.3, Good agent ~0.7–0.85

**Why it's easy:**
- Single goal to optimize (athletic performance)
- Short episode (14 days)
- High-compliance persona (70%)
- Athletic performance persona is responsive to exercise interventions

### 6.3 Task 2: Multi-Outcome Balance (Medium) — `multi_outcome`

**Configuration:**
| Parameter | Value |
|---|---|
| Persona | `stress_management` |
| Goal | `STRESS_MANAGEMENT` |
| Compliance rate | 0.5 |
| Episode length | 30 days |

**Grader:**
```
Score = 0.35 * normalize(avg_reward, 40, 75)
     + 0.25 * biomarker_breadth (fraction of 8 markers that improved)
     + 0.20 * consistency (1 - normalize(stddev(rewards), 0, 15))
     + 0.20 * max(0, reward_trend)
```
- 35% average reward (overall improvement quality)
- 25% **biomarker breadth** (how many of 8 biomarkers improved — rewards balanced improvement, not tunnel-vision on one metric)
- 20% consistency (stable improvement, not erratic)
- 20% reward trend (improving over time)
- Expected scores: Random ~0.15–0.25, Good agent ~0.5–0.65

**Why it's medium:**
- Must balance all 8 biomarkers, not just one
- Stress management persona has poor starting biomarkers (high cortisol, low HRV, low VO2)
- Lower compliance (50%) means more adaptation needed
- 30-day episode requires sustained strategy

### 6.4 Task 3: Resistant User Adaptation (Hard) — `resistant_adaptation`

**Configuration:**
| Parameter | Value |
|---|---|
| Persona | `weight_loss` |
| Goal | `WEIGHT_LOSS` |
| Compliance rate | 0.25 |
| Episode length | 30 days |

**Grader:**
```
Score = 0.25 * normalize(avg_reward, 35, 65)
     + 0.25 * normalize(improvement_last7_vs_first7, 0, 15)
     + 0.20 * consistency
     + 0.15 * compliance_effectiveness
     + 0.15 * biomarker_breadth
```
- 25% average reward (against a harder baseline — poor starting biomarkers)
- 25% **outcome improvement** (last 7 days avg reward vs first 7 days — did the agent make things better?)
- 20% consistency (stable improvement, not erratic)
- 15% **compliance effectiveness** (did the agent achieve actual compliance above the configured 25%? Measures whether it adapted recommendations to be more followable)
- 15% biomarker breadth (balanced improvement across markers)
- Expected scores: Random ~0.05–0.15, Good agent ~0.35–0.50

**Why it's hard:**
- 75% of recommendations are ignored
- Agent must learn gradual, small-step strategies that a resistant user will actually follow
- Weight loss persona has poor starting fitness (low VO2, high body fat, overtraining threshold of only 2 days)
- Compliance effectiveness grading explicitly rewards understanding the user's resistance
- Improvement metric means the agent can't just coast — must show delta over time

---

## 7. Personas

Fixed set of personas for Round 1. Each defines starting biomarkers, default behavior, compliance rate, health goal, and a **hidden physiological response model**.

### 7.1 Persona Overview

| Persona | Goal | Compliance | Default Sleep | Default Exercise | Default Nutrition |
|---|---|---|---|---|---|
| `athletic_performance` | Athletic Performance | 0.7 | SHORT (6-7h) | HIIT | High Protein |
| `stress_management` | Stress Management | 0.5 | VERY_SHORT (<6h) | None | Processed |
| `weight_loss` | Weight Loss | 0.25 | SHORT (6-7h) | None | Processed |

### 7.2 Starting Biomarkers

| Biomarker | Athletic Performance | Stress Management | Weight Loss |
|---|---|---|---|
| Resting HR (bpm) | 62 | 78 | 82 |
| HRV (ms) | 55 | 30 | 25 |
| VO2 Max (ml/kg/min) | 42 | 30 | 25 |
| Body Fat (%) | 18 | 28 | 35 |
| Lean Mass (kg) | 65 | 55 | 58 |
| Sleep Efficiency (%) | 78 | 65 | 70 |
| Cortisol Proxy | 35 | 75 | 55 |
| Energy Level | 70 | 35 | 40 |

### 7.3 Hidden Response Model Differences

The key insight is that each persona responds differently to the same actions:

| Trait | Athletic Performance | Stress Management | Weight Loss |
|---|---|---|---|
| VO2 cardio gain | 0.25 (high) | 0.12 (moderate) | 0.08 (low) |
| Lean mass strength gain | 0.08 (high) | 0.04 (moderate) | 0.03 (low) |
| Overtraining threshold | 3 days | 3 days | 2 days (fragile) |
| HRV sleep sensitivity | 3.0 | 5.0 (very responsive) | 3.5 |
| Cortisol exercise stress | 5.0 | 8.0 (high) | 10.0 (very high) |
| Protein recovery multiplier | 1.4 | 1.3 | 1.3 |
| Sleep debt exercise penalty | 0.1 | 0.1 | 0.15 (severe) |

**What this means for agents:**
- The athletic performance persona gains VO2 max quickly from cardio but is sensitive to overtraining
- The stress management persona benefits enormously from sleep and yoga (cortisol/HRV response) but gets stressed from intense exercise
- The weight loss persona has a low overtraining threshold, high cortisol sensitivity to exercise, and severe sleep-debt penalties — requiring gentle, gradual interventions

### 7.4 Persona Default Actions

When the simulated user doesn't comply with a recommendation, they fall back to persona defaults:

- **Athletic Performance:** Defaults to short sleep, HIIT, high-protein — generally an active person who under-sleeps
- **Stress Management:** Defaults to very short sleep, no exercise, processed food — the worst defaults
- **Weight Loss:** Defaults to short sleep, no exercise, processed food — and resists change on all levers

---

## 8. Inference Script (`inference.py`)

The baseline agent uses an LLM via the OpenAI API client.

### 8.1 Requirements

- Located at project root: `inference.py`
- Uses env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` (also accepts `OPENAI_API_KEY`)
- Runs all 3 tasks sequentially
- Structured stdout logging (see §8.2)
- Must complete all tasks within 20 minutes total
- Uses the OpenAI Python client (`openai` package)

### 8.2 Stdout Format

Each task emits a `[START]` header, per-step `[STEP]` lines, and a `[END]` summary. The format is designed so that graders, judges, and post-hoc analysis can reconstruct the full episode from stdout alone.

**`[START]` — task header (emitted once per task):**
```
[START] task=single_goal env=wellness-outcome model=<MODEL_NAME> persona=athletic_performance goal=athletic_performance compliance=0.7 days=14
```

| Field | Purpose |
|---|---|
| `task` | Task name being run |
| `env` | Environment identifier |
| `model` | LLM model name (from env var) |
| `persona` | Active persona for this task |
| `goal` | User's health goal |
| `compliance` | Configured compliance rate (how often the user follows recommendations) |
| `days` | Total episode length |

**`[STEP]` — per-day trace (emitted every step):**
```
[STEP] step=1 action={"sleep":"7_to_8h","exercise":"moderate_cardio","nutrition":"balanced"} actual={"sleep":"6_to_7h","exercise":"none","nutrition":"balanced"} complied=false biomarkers={"resting_hr":61.5,"hrv":56.2,"vo2_max":42.1,...} deltas={"resting_hr":-0.5,"hrv":+1.2,...} reward=58.30 done=false error=null
```

| Field | Purpose | Differentiating Signal |
|---|---|---|
| `step` | Current day (1-indexed) | — |
| `action` | LLM-recommended action (JSON) | Shows agent strategy — are recommendations adapting to the persona? |
| `actual` | What the simulated user actually did (JSON) | Reveals compliance gaps — did the user follow the recommendation? |
| `complied` | Whether the user followed the recommendation (`true`/`false`) | Directly tracks compliance; judges can see adaptation patterns |
| `biomarkers` | Current biomarker values after this step (JSON) | Shows actual health trajectory — are outcomes improving? |
| `deltas` | Biomarker changes this step (JSON) | The raw outcome signal — which biomarkers improved/declined? |
| `reward` | Total reward for this step (0–100, baseline 50) | Reward trajectory is key: values >50 = biomarkers improving |
| `done` | Episode complete? | — |
| `error` | Error message if step failed, else `null` | Any non-null errors signal fragility |

**`[END]` — task summary (emitted once per task):**
```
[END] task=single_goal success=true steps=14 score=0.72 avg_reward=58.40 reward_trend=+1.05 reward_stddev=4.30 compliance_rate_actual=0.64 rewards=58.30,55.20,...
```

| Field | Purpose | Differentiating Signal |
|---|---|---|
| `task` | Task name | — |
| `success` | Whether the task completed without fatal errors | `false` = zero credit |
| `steps` | Total steps taken | Should match `days` from `[START]` |
| `score` | **Grader output (0.0–1.0)** | **The headline metric.** |
| `avg_reward` | Mean reward across all steps | Overall outcome quality (>50 = net improvement) |
| `reward_trend` | Linear slope of rewards over the episode | Positive = agent learned to improve outcomes over time |
| `reward_stddev` | Standard deviation of rewards | Low = consistent strategy; high = erratic |
| `compliance_rate_actual` | Fraction of steps where user actually complied | Divergence from configured rate reveals if agent adapted |
| `rewards` | Comma-separated per-step rewards | Full trajectory for post-hoc analysis |

### 8.2.1 Why These Fields Matter for Scoring

The stdout fields directly map to grader components:

| Grader Component | Weight (varies by task) | Stdout Fields That Reveal It |
|---|---|---|
| Average reward (biomarker outcomes) | 25–60% | `reward` per step, `avg_reward` in `[END]` |
| Biomarker breadth | 15–25% | `biomarkers` across steps — how many improved start to end? |
| Consistency (low variance) | 20% | `reward_stddev` in `[END]`, stability of `reward` across `[STEP]`s |
| Trend (improvement over time) | 20% | `reward_trend` in `[END]`, ascending `reward` in `[STEP]`s |
| Compliance effectiveness (Task 3) | 15% | `complied` across steps — does actual compliance exceed configured rate? |

**A strong submission's stdout will show:**
1. **Rewards consistently above 50** — biomarkers are improving, not declining
2. **Ascending rewards** — reward values trend upward across steps
3. **Biomarker breadth** — `biomarkers` show improvement across many markers, not just one
4. **Low variance** — no wild swings in reward between consecutive steps
5. **For Task 3:** `compliance_rate_actual` exceeds the configured 0.25 because the agent's gentler recommendations are more followable

### 8.2.2 Example: Strong vs Weak Agent Comparison

**Weak agent (Task 3 — resistant_adaptation):**
```
[STEP] step=1  action={"sleep":"8_to_9h","exercise":"hiit","nutrition":"high_protein"} actual={"sleep":"6_to_7h","exercise":"none","nutrition":"processed"} complied=false biomarkers={"resting_hr":82.5,...} deltas={"resting_hr":+0.5,...} reward=42.0 ...
[STEP] step=2  action={"sleep":"8_to_9h","exercise":"hiit","nutrition":"high_protein"} actual={"sleep":"6_to_7h","exercise":"none","nutrition":"processed"} complied=false biomarkers={"resting_hr":83.0,...} deltas={"resting_hr":+0.5,...} reward=41.5 ...
...
[END] task=resistant_adaptation success=true steps=30 score=0.08 avg_reward=42.50 reward_trend=-0.10 reward_stddev=2.10 compliance_rate_actual=0.23 rewards=42.0,41.5,...
```
*Problem: Agent keeps recommending extreme changes → user ignores all of them → biomarkers decline → rewards stay below 50 → low score.*

**Strong agent (Task 3 — resistant_adaptation):**
```
[STEP] step=1  action={"sleep":"6_to_7h","exercise":"light_cardio","nutrition":"high_carb"} actual={"sleep":"6_to_7h","exercise":"none","nutrition":"high_carb"} complied=false biomarkers={"resting_hr":81.8,...} deltas={"resting_hr":-0.2,...} reward=51.50 ...
[STEP] step=5  action={"sleep":"7_to_8h","exercise":"yoga","nutrition":"balanced"} actual={"sleep":"6_to_7h","exercise":"yoga","nutrition":"balanced"} complied=false biomarkers={"resting_hr":80.5,...} deltas={"resting_hr":-0.3,...} reward=55.80 ...
[STEP] step=25 action={"sleep":"7_to_8h","exercise":"moderate_cardio","nutrition":"balanced"} actual={"sleep":"7_to_8h","exercise":"light_cardio","nutrition":"balanced"} complied=false biomarkers={"resting_hr":78.0,...} deltas={"resting_hr":-0.2,...} reward=58.30 ...
...
[END] task=resistant_adaptation success=true steps=30 score=0.42 avg_reward=55.80 reward_trend=+0.85 reward_stddev=3.20 compliance_rate_actual=0.40 rewards=51.50,52.20,...,58.30
```
*Key: Agent starts with small, achievable recommendations → user partially complies → biomarkers gradually improve → rewards trend upward → compliance_rate_actual (0.40) exceeds configured 0.25.*

### 8.3 Agent Strategy (Baseline)

The LLM agent receives the observation as JSON, the action space description, and a system prompt explaining the outcome-based wellness domain. It returns a structured `Action`.

**System prompt should include:**
- The 8 biomarkers and what direction is "good" for each
- The user's health goal and what it prioritizes
- The persona's compliance rate and what it means
- Current biomarker values and recent deltas/trends
- Instruction to give gradual recommendations for low-compliance personas
- Output format: JSON matching the `Action` model

---

## 9. Project Structure

```
wellness-simulator/
├── openenv.yaml                   # OpenEnv metadata
├── Dockerfile                     # HF Spaces deployment
├── requirements.txt               # Python dependencies
├── inference.py                   # Baseline LLM agent
├── README.md                      # Project overview + task descriptions
├── wellness_env/
│   ├── __init__.py                # Exports WellnessEnv, models
│   ├── env.py                     # WellnessEnv class (reset/step/state)
│   ├── models.py                  # Pydantic models (Action, Observation, etc.)
│   ├── simulator.py               # Transition dynamics (energy, fatigue, debt)
│   ├── payoff.py                  # Payoff function (biomarker delta rewards)
│   ├── personas.py                # Persona definitions and compliance model
│   └── graders.py                 # Task graders (score 0.0–1.0)
└── tests/
    ├── test_env.py                # OpenEnv interface compliance tests
    ├── test_payoff.py             # Payoff function unit tests
    ├── test_simulator.py          # Simulator transition tests
    └── test_graders.py            # Grader determinism + range tests
```

### 9.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference.py"]
```

### 9.2 Dependencies (`requirements.txt`)

```
pydantic>=2.0
openai>=1.0
numpy
```

No heavy ML frameworks needed — the environment is pure Python math. The agent uses the OpenAI client.

---

## 10. Validation Checklist

Before submission, verify:

- [ ] `openenv validate` passes
- [ ] `docker build -t wellness-simulator .` succeeds
- [ ] `docker run wellness-simulator` completes in <20 minutes
- [ ] All 3 tasks produce `[START]`/`[STEP]`/`[END]` stdout
- [ ] Graders return scores in [0.0, 1.0] range
- [ ] Graders are deterministic (same seed → same score)
- [ ] `reset()` returns valid `Observation`
- [ ] `step()` returns valid `StepResult`
- [ ] `state()` returns valid `EnvState`
- [ ] Random agent scores lower than LLM agent on all tasks
- [ ] HF Space deploys and responds to health check

---

## 11. Scoring Expectations

| Task | Random Agent | Baseline LLM | Good LLM | Theoretical Max |
|---|---|---|---|---|
| single_goal (easy) | 0.20–0.30 | 0.50–0.65 | 0.70–0.85 | ~0.90 |
| multi_outcome (medium) | 0.15–0.25 | 0.35–0.45 | 0.50–0.65 | ~0.75 |
| resistant_adaptation (hard) | 0.05–0.15 | 0.20–0.30 | 0.35–0.50 | ~0.55 |

The theoretical max is capped by compliance rates and the hidden response model — even a perfect strategy can't overcome a user who ignores 75% of recommendations and has a fragile physiology.

---

## 12. Hackathon Judging Alignment

| Judging Criterion | Weight | How This Environment Addresses It |
|---|---|---|
| Real-world utility | 30% | Wellness optimization is a universal need; outcome-based biomarker tracking models real physiology; personas model real user behavior including non-compliance |
| Task & grader quality | 25% | 3 tasks with clear difficulty progression; graders measure outcome improvement, consistency, trend, breadth, and compliance adaptation |
| Environment design | 20% | Outcome-based reward from biomarker deltas; hidden physiological response models per persona; compliance modeling; stochasticity |
| Code quality & spec compliance | 15% | Pydantic models, OpenEnv interface, structured project, tests |
| Creativity & novelty | 10% | Compliance adaptation grading; persona-based difficulty; multi-outcome trade-offs |

---

## 13. Relationship to Full Vision

This Round 1 submission implements a **subset** of the full requirements:

| Full Doc Section | Round 1 Usage |
|---|---|
| §1 Vision | Outcome-based biomarker rewards |
| §2 Architecture | Simplified — environment only, no app |
| §3 Payoff Function | Implemented as `payoff.py` with goal-weighted biomarker delta scoring |
| §4 RL Environment | Implemented as OpenEnv with hidden response models per persona. LLM agent, not trained NN. |
| §5 User App | Skipped |
| §6 Multi-Agent Demo | Personas reused for task difficulty; visualization deferred |
| §7 Versioning | Single version (V1) |
| §8 Tech Stack | Minimal: Python, Pydantic, OpenAI client, NumPy |
| §9 Observability | Structured stdout logging only |
| §10 Design Principles | Outcome-driven, interpretable, modular |
| §13 Privacy | Not applicable (simulated users only) |
