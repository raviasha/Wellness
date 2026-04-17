# Architecture — Wellness Outcome

This document describes the system architecture, module responsibilities, data flow, and key design decisions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                       │
│                                                                             │
│  Next.js Dashboard (static)    OpenEnv Evaluator     iOS Shortcuts          │
│  ├── GarminDashboard.tsx       ├── inference.py      └── Apple Health       │
│  ├── AdminPanel.tsx            │   POST /reset           POST /api/health/  │
│  ├── ManualLogForm.tsx         │   POST /step             apple-push        │
│  └── UserManual.tsx            │   GET  /grade                              │
│         │                      │       │                       │            │
└─────────┼──────────────────────┼───────┼───────────────────────┼────────────┘
          │ fetch()              │ HTTP  │                       │
          ▼                      ▼       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         app.py  (FastAPI)                                    │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ OpenEnv  │ │  Users   │ │ Wearable │ │Nutrition │ │  Training +      │  │
│  │ /reset   │ │ /api/    │ │ /api/    │ │ /api/    │ │  Inference       │  │
│  │ /step    │ │ users/   │ │ garmin/  │ │nutrition/│ │  /api/train      │  │
│  │ /state   │ │ profile  │ │ wearable/│ │ logs/    │ │  /api/recommend  │  │
│  │ /grade   │ │ device   │ │ terra/   │ │          │ │  /api/calibrate  │  │
│  │ /health  │ │          │ │ upload   │ │          │ │  /api/persist    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │             │            │             │                │            │
│       ▼             ▼            ▼             ▼                ▼            │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     Backend Service Layer                            │    │
│  │  database.py  garmin_service.py  terra_service.py  upload_service.py│    │
│  │  inference_service.py  distribution_calibration.py  eval_service.py │    │
│  │  llm_nutrition.py  persist.py                                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     Core Domain Layer                                │    │
│  │  wellness_env/env.py  simulator.py  distribution_simulator.py       │    │
│  │  payoff.py  graders.py  personas.py  models.py                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     Storage                                          │    │
│  │  SQLite (wellness.db)    PyTorch models (models/user_{id}/*.pt)     │    │
│  │  HuggingFace Space repo (persist.py → git push)                     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

### 1. Entry Points

| File | Role | When Used |
|------|------|-----------|
| `app.py` | FastAPI server, ~40 endpoints, serves SPA | Always (production + hackathon) |
| `inference.py` | OpenEnv LLM agent, structured stdout | Hackathon evaluation only |
| `run_inference.py` | Loads `.env` then calls `inference.main()` | Dev convenience |

### 2. Wellness Environment (`wellness_env/`)

The core RL environment, independent of the production app.

```
wellness_env/
├── models.py         ← Pydantic types: Action, Biomarkers, StepResult, EnvState, Goal
├── personas.py       ← 4 archetypes (athletic, stress, weight_loss, digital_twin)
│                        each with hidden ResponseModel (~15 physiological params)
├── simulator.py      ← Rule-based response: action + persona params → biomarker Δ
│                        History effects: sleep debt, overtraining, adaptation
├── distribution_simulator.py  ← Gaussian copula alternative (uses real calibration data)
├── payoff.py         ← Reward: R = 50 + Σ(normalize(Δᵢ) × wᵢ × 100), clamped [0,100]
│                        3 goal weight profiles × 8 biomarkers
├── graders.py        ← Task-specific 0.0–1.0 scoring (avg_reward, trend, consistency)
├── env.py            ← WellnessEnv: reset/step/state/grade interface
│                        4 task configurations with different persona/duration/difficulty
└── __init__.py
```

**Key design:** The `ResponseModel` in each persona acts as a hidden physiological "truth" — the agent must discover the optimal policy through interaction, not by inspecting the model.

### 3. Backend Services (`backend/`)

Production-specific services that don't exist in the hackathon environment.

```
backend/
├── database.py                 ← SQLAlchemy ORM (User, UserProfile, WearableSync,
│                                  ManualLog, Recommendation). 6 migration steps.
│                                  Smart merge: partial syncs don't overwrite good data.
│
├── inference_service.py        ← THE BRAIN. 4 fidelity levels:
│                                  L0: Generic LLM  →  L1: LLM + history
│                                  L2: NN + LLM (calibrated)  →  L3: NN + LLM (trained)
│                                  Builds 14-day causal history ledger (T-1 → T alignment)
│
├── distribution_calibration.py ← Gaussian copula: fits P(actions, outcomes) joint
│                                  distribution with Ledoit-Wolf shrinkage (n=15-30),
│                                  samples P(outcomes | actions) for predictions
│
├── eval_service.py             ← Compares yesterday's recommendation with today's actuals
│                                  fidelity_score = 60% magnitude + 40% directional accuracy
│                                  Per-input compliance (sleep/exercise/nutrition)
│
├── garmin_service.py           ← Garmin Connect API: 3-tier auth (cache → tokens → login)
│                                  8 data categories with per-category error isolation
│
├── terra_service.py            ← Terra API: OAuth widget → webhooks → polling fallback
│                                  Normalizes Apple Watch / Fitbit → WearableSync columns
│
├── upload_service.py           ← Apple Health XML / CSV / JSON parser
│                                  60+ column name aliases, derived field computation
│
├── llm_nutrition.py            ← GPT-4o-mini: "chicken breast 200g" → {cal, protein, ...}
│                                  Also handles food dedup decisions (overwrite vs append)
│
├── persist.py                  ← Uploads wellness.db + models/ to HF Space git repo
│                                  Survives container restarts
│
├── calibration.py              ← ⚠️ SUPERSEDED by distribution_calibration.py (OLS-based)
├── evals.py                    ← ⚠️ SUPERSEDED by eval_service.py
├── guardian.py                  ← ⚠️ DEAD CODE (process supervisor, unused in Docker)
└── calibrated_persona.json     ← Example output from calibration
```

### 4. RL Training (`rl_training/`)

```
rl_training/
├── ppo_lite.py     ← ActorCritic: 2×128 hidden (Tanh), separate actor/critic heads
│                      PPO: clipped objective, GAE (λ=0.95, γ=0.99)
├── env_wrapper.py  ← Gymnasium adapter: 150 discrete actions (5×6×5) → flat 19-float obs
│                      Action = (sleep_idx, exercise_idx, nutrition_idx)
├── train.py        ← Training loop: 50k steps, saves to models/user_{id}/
│                      Uses distribution_simulator when calibration data available
└── __init__.py
```

### 5. Frontend (`webapp/`)

Next.js 16 app with static export. No SSR or API routes live in the frontend — everything goes through `app.py`.

```
webapp/src/
├── app/
│   ├── layout.tsx       ← Root layout: Geist fonts, PWA meta, service worker registration
│   ├── page.tsx         ← Renders <GarminDashboard />
│   ├── globals.css      ← Dark theme: #0a0a0f bg, indigo/violet accents, glassmorphism
│   ├── page.module.css
│   └── favicon.ico
└── components/
    ├── GarminDashboard.tsx  ← Main dashboard (1200 lines). Multi-user switcher, tabs for
    │                           dashboard/manual-log/settings/admin/evals. Recharts for
    │                           8 behavioral input sparklines + 5 outcome charts. AI coach
    │                           panel with expected deltas + healthspan impact text.
    ├── AdminPanel.tsx       ← All-users overview with sync/log data
    ├── ManualLogForm.tsx    ← Food/weight/note logging with LLM macro parsing
    └── UserManual.tsx       ← In-app help text
```

---

## Data Flow — Production Recommendation Cycle

```
Day T                                    Day T+1
─────                                    ───────

User wears device                        User wears device
        │                                        │
        ▼                                        ▼
  Wearable sync                            Wearable sync
  (Garmin/Terra/Upload)                    (Garmin/Terra/Upload)
        │                                        │
        ▼                                        ▼
  WearableSync row (day T)               WearableSync row (day T+1)
  ┌─────────────────────┐                ┌─────────────────────┐
  │ hrv, rhr, sleep,    │                │ hrv, rhr, sleep,    │
  │ stress, recovery,   │                │ stress, recovery,   │
  │ spo2, steps,        │                │ spo2, steps,        │
  │ active_cal, etc.    │                │ active_cal, etc.    │
  └─────────┬───────────┘                └─────────┬───────────┘
            │                                      │
            │  ┌─────────────────┐                 │
            └──► inference_      │                 │
               │ service.py      │                 │
               │                 │                 │
               │ 1. Build causal │                 │
               │    history      │                 │
               │    (14-day      │                 │
               │     ledger of   │                 │
               │     input@T-1   │                 │
               │     → output@T) │                 │
               │                 │                 │
               │ 2. Determine    │                 │
               │    fidelity     │                 │
               │    level (0-3)  │                 │
               │                 │                 │
               │ 3. NN forward   │                 │
               │    pass (if L2+)│                 │
               │                 │                 │
               │ 4. Distribution │                 │
               │    simulator →  │                 │
               │    expected Δ   │                 │
               │                 │                 │
               │ 5. LLM synth   │                 │
               │    → coaching   │                 │
               │    text         │                 │
               └────────┬────────┘                 │
                        │                          │
                        ▼                          │
               Recommendation                     │
               ┌──────────────────┐                │
               │ sleep: "long"    │                │
               │ exercise: "mod"  │                │
               │ nutrition: "good"│                │
               │ expected_Δ_hrv:  │                │
               │   +3.2ms         │                │
               │ long_term: "..."│                │
               └────────┬─────────┘                │
                        │                          │
                        │    Next day              │
                        │    eval_service.py ◄─────┘
                        │         │
                        ▼         ▼
                   fidelity_score = compare(expected_Δ, actual_Δ)
                   compliance = { sleep: 0.8, exercise: 0.6, nutrition: 0.9 }
```

---

## Data Flow — OpenEnv Evaluation

```
inference.py
    │
    ├── POST /reset { task_name: "single_goal" }
    │         │
    │         ▼
    │   WellnessEnv.reset()
    │   ├── Select persona (athletic)
    │   ├── Initialize biomarkers from persona defaults
    │   └── Return EnvState { observation, task_description }
    │
    ├── loop for max_steps:
    │   │
    │   ├── LLM decides action (or fallback to rules)
    │   │
    │   ├── POST /step { action: { sleep, exercise, nutrition } }
    │   │         │
    │   │         ▼
    │   │   WellnessEnv.step()
    │   │   ├── personas.apply_compliance(action, persona)
    │   │   ├── simulator.compute_biomarker_changes(action, biomarkers, persona)
    │   │   ├── payoff.compute_reward(old, new, goal)
    │   │   └── Return StepResult { observation, reward, done }
    │   │
    │   └── stdout: [STEP] step=N action={...} reward=R done=D
    │
    └── GET /grade → graders.grade(task, history) → 0.0–1.0
              stdout: [END] success=true score=S
```

---

## Database Schema

```sql
User
├── id (PK)
├── username (unique)
├── garmin_email / garmin_password_enc    -- Direct Garmin auth
├── terra_user_id                         -- Terra API user
├── wearable_source                       -- "garmin" | "terra" | "upload"
└── created_at

UserProfile
├── user_id (FK → User)
├── name, age, weight_kg, height_cm
├── goal                                  -- "improve_fitness" | "reduce_stress" | ...
├── compliance_rate
├── simulator_approved                    -- Boolean: user approved calibrated persona
├── last_calibration_at
└── simulator_params_json                 -- Serialized JointDistribution

WearableSync
├── id (PK)
├── user_id (FK → User)
├── sync_date (unique per user per day)
├── source                                -- "garmin" | "terra" | "upload" | "simulate"
├── hrv_avg, rhr, sleep_hours, sleep_score, stress_avg
├── recovery_score, spo2_avg, vo2_max, body_battery_high/low
├── steps, active_calories, intensity_minutes_moderate/vigorous
├── respiration_avg, body_fat_pct, weight_kg
├── calories_in, protein_g, carbs_g, fat_g, nutrition_quality
└── created_at

ManualLog
├── id (PK)
├── user_id (FK → User)
├── log_date, log_time
├── log_type                              -- "food" | "weight" | "exercise" | "note"
├── value, raw_input
└── created_at

Recommendation
├── id (PK)
├── user_id (FK → User)
├── rec_date
├── expected_delta_{hrv,rhr,sleep,stress,weight}
├── actual_delta_{hrv,rhr,sleep,stress,weight}
├── compliance_{sleep,exercise,nutrition}
├── fidelity_score
├── long_term_impact
└── created_at
```

---

## API Endpoint Map

### OpenEnv Spec

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/reset` | `reset()` | Start new episode |
| POST | `/step` | `step()` | Submit action, get reward |
| GET | `/state` | `state()` | Current env state |
| GET | `/grade` | `grade()` | Final task score |
| GET | `/health` | `health()` | Liveness |

### User Management

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/users` | `list_users()` | List all users |
| POST | `/api/users/create` | `create_user()` | Create user |
| GET/PUT | `/api/profile` | `get/save_profile()` | User health profile |
| POST | `/api/users/device` | `update_device()` | Set wearable source + credentials |

### Wearable Data

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/garmin/sync` | `garmin_sync()` | Sync from Garmin Connect |
| POST | `/api/wearable/sync` | `wearable_sync()` | Universal sync (routes by source) |
| POST | `/api/wearable/upload` | `wearable_upload()` | Upload Apple Health XML/CSV/JSON |
| POST | `/api/health/apple-push` | `apple_push()` | iOS Shortcuts live push |
| POST | `/api/terra/connect` | `terra_connect()` | Generate Terra OAuth widget session |
| POST | `/api/terra/webhook` | `terra_webhook()` | Receive Terra webhook payloads |

### Nutrition & Logging

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/nutrition/parse` | `parse_nutrition()` | LLM food → macros |
| POST | `/api/logs/manual` | `manual_log()` | Save manual log entry |

### Calibration & Training

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/calibrate` | `calibrate()` | Fit Gaussian copula from history |
| GET | `/api/persona/draft` | `persona_draft()` | Preview calibrated persona |
| POST | `/api/persona/approve` | `persona_approve()` | Lock in persona for training |
| POST | `/api/train` | `train()` | Start PPO training (async) |
| GET | `/api/train/status` | `train_status()` | Training progress |

### Inference & Evaluation

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/recommendations` | `recommendations()` | Get hybrid NN+LLM coaching |
| GET | `/api/persona/evals` | `persona_evals()` | Eval history for user |
| GET | `/api/dashboard/metrics` | `dashboard_metrics()` | Aggregated dashboard stats |

### Admin

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET/PUT | `/api/admin/config` | `admin_config()` | Runtime config |
| GET | `/api/admin/data` | `admin_data()` | Bulk data export |
| POST | `/api/persist` | `persist()` | Upload DB + models to HF |

---

## Key Design Decisions

### 1. Causal Lag Alignment (T-1 → T)

Behavioral inputs (sleep, exercise, nutrition) from day T are matched with biological outcomes (HRV, recovery, stress) from day T+1. This correctly models the ~24-hour delay in physiological adaptation. The inference service builds a "causal history ledger" with this alignment.

### 2. Smart Merge on Sync

When a wearable sync arrives with partial data (e.g., Garmin returns HRV but not sleep), the system only updates non-null fields. This prevents partial syncs from overwriting existing high-fidelity data.

### 3. Ledoit-Wolf Shrinkage for Small Samples

With only 15-30 calibration points, the raw covariance matrix would be near-singular. Ledoit-Wolf shrinkage regularizes toward a structured target, making conditional sampling stable.

### 4. 4-Level Fidelity Ladder

The system gracefully degrades: with zero data it gives generic LLM coaching; with some history it personalizes via context; with enough data it calibrates a statistical model; with a trained NN it optimizes directly. This means the app is useful from day one.

### 5. Static Export + FastAPI Serving

The Next.js app is exported as static HTML/JS/CSS (no Node.js runtime needed in production). FastAPI serves these files via a catch-all route, eliminating the need for a separate web server.

---

## Security Considerations

| Area | Status | Risk |
|------|--------|------|
| SQL Injection | Protected | SQLAlchemy parameterized queries |
| Credential Storage | Partial | Fernet encryption, but hardcoded fallback key exists |
| API Authentication | Missing | `X-User-Id` header trusted without verification |
| Terra Webhooks | Protected | HMAC signature verification implemented |
| CORS | Open | `allow_origins=["*"]` — acceptable for HF Spaces |
| Rate Limiting | Missing | No protection against API abuse |

---

## Files Flagged for Cleanup

### Safe to Delete (dead code / duplicates)

| File | Lines | Why |
|------|-------|-----|
| `hf_deploy/` | ~entire project | Full deployment copy — use Docker build instead |
| `hf_deploy 2/` | ~entire project | Another deployment copy |
| `backend/guardian.py` | ~30 | Process supervisor — Docker handles restarts |
| `backend/evals.py` | ~130 | Superseded by `eval_service.py` |
| `backend/calibration.py` | ~165 | Superseded by `distribution_calibration.py` |
| `demo.py` | ~320 | Chart generator, not used by app |
| `demo_output/` | directory | Generated charts |
| `deploy_copy.py` | ~30 | Manual copy script |
| `hf_upload.py` | ~50 | Manual upload script |
| `scratch/` | directory | Experimental scripts |
| `payload.json` | — | Dev artifact (raw Garmin response) |
| `payload2.json` | — | Dev artifact |
| `format2.json` | — | Dev artifact |
| `formatted_payload.json` | — | Dev artifact |
| `export_cda.xml` | — | Dev artifact |
| `static/_next 2/` | directory | Stale build artifact |
| `static/_next 3/` | directory | Stale build artifact |
| `guardian.log` | — | Log file (should be gitignored) |
| `guardian_crash.log` | — | Log file |
| `server.log` | — | Log file |

### Refactor Targets

| Priority | Target | Current | Proposed |
|----------|--------|---------|----------|
| High | `app.py` | 1115-line monolith with ~40 endpoints | Split into 6-7 FastAPI routers |
| High | Auth | No authentication | Add JWT or session-based auth middleware |
| High | Fernet key | Hardcoded fallback `FALLBACK_KEY` | Fail-fast if `FERNET_KEY` not set |
| Medium | `GarminDashboard.tsx` | 1200-line mega-component | Extract `BiometricCharts`, `RecommendationPanel`, `SyncControls` |
| Medium | `database.py` `add_manual_log()` | LLM call inside DB layer | Move food dedup logic to service layer |
| Medium | `static/` in git | ~9M of build artifacts tracked | Gitignore, generate during Docker build |
| Low | `_sync_backoff` | In-memory dict (lost on restart) | Use DB or Redis for rate limit state |
| Low | `Goal` enum | 3 values in model, more referenced in code | Align enum with actual goals used |

---

## Dependency Graph

```
app.py
├── wellness_env.env         (OpenEnv endpoints)
├── wellness_env.models       (Pydantic types everywhere)
├── backend.database          (all CRUD operations)
├── backend.garmin_service    (wearable sync)
├── backend.terra_service     (wearable sync)
├── backend.upload_service    (file upload)
├── backend.llm_nutrition     (food parsing)
├── backend.inference_service (recommendations)
│   ├── backend.distribution_calibration
│   ├── wellness_env.distribution_simulator
│   ├── rl_training.ppo_lite
│   └── openai (GPT-4o-mini)
├── backend.eval_service      (daily eval)
├── backend.calibration       ← REDUNDANT (also imports distribution_calibration)
├── backend.persist           (HF persistence)
└── rl_training.train         (async training)
    ├── rl_training.ppo_lite
    └── rl_training.env_wrapper
        └── wellness_env.env
```

---

## Test Coverage Map

```
wellness_env/
├── env.py                  ✅ test_env.py
├── models.py               ✅ test_env.py (via model validation)
├── simulator.py            ✅ test_simulator.py
├── distribution_simulator.py ✅ test_distribution_simulator.py
├── graders.py              ✅ test_graders.py
├── payoff.py               ✅ test_payoff.py
└── personas.py             ✅ test_simulator.py (implicit)

backend/
├── database.py             ✅ test_database_persistence.py
├── distribution_calibration.py ✅ test_distribution_calibration.py
├── llm_nutrition.py        ✅ test_llm_nutrition.py (requires API key)
├── inference_service.py    ❌ NO TESTS
├── eval_service.py         ❌ NO TESTS
├── garmin_service.py       ❌ NO TESTS
├── terra_service.py        ❌ NO TESTS
├── upload_service.py       ❌ NO TESTS
└── persist.py              ❌ NO TESTS

rl_training/
├── ppo_lite.py             ❌ NO TESTS
├── env_wrapper.py          ❌ NO TESTS (indirectly via test_env)
└── train.py                ❌ NO TESTS

app.py (API endpoints)      ❌ NO TESTS
webapp/                      ❌ NO TESTS
```
