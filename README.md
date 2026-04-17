---
title: Wellness Outcome Digital Twin
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Wellness Outcome — Autonomous AI Health Coach

An **outcome-based wellness coaching platform** that builds a personal "digital twin" from real wearable data, trains a per-user PPO neural network to learn your body's response patterns, and delivers hybrid NN + LLM recommendations through a dark-themed Next.js dashboard.

Built for the **Scaler × Meta PyTorch Hackathon** as an [OpenEnv](https://github.com/OpenEnvs/openenv-core)-compliant real-world RL environment.

**Live:** [raviasha-wellness-outcome.hf.space](https://raviasha-wellness-outcome.hf.space)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [OpenEnv Interface](#openenv-interface)
- [Environment Design](#environment-design)
- [Production Coaching System](#production-coaching-system)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Tests](#tests)
- [Refactoring Notes](#refactoring-notes)

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│                          DATA COLLECTION                              │
│                                                                       │
│   Garmin Connect ──┐                                                  │
│   Apple Watch ─────┼── Terra API / Direct Scraping / CSV Upload ──┐   │
│   Fitbit ──────────┘                                              │   │
│                                                                   ▼   │
│                                                    ┌──────────────┐   │
│   Manual Logs ─── Food / Weight / Notes ──────────►│  SQLite DB   │   │
│   (LLM-parsed)    via GPT-4o-mini                  │ (WearableSync│   │
│                                                    │  ManualLog)  │   │
│                                                    └──────┬───────┘   │
├───────────────────────────────────────────────────────────┼───────────┤
│                       CALIBRATION (≥15 days)              │           │
│                                                           ▼           │
│   Gaussian Copula Regression ── Ledoit-Wolf shrinkage ──► Personal   │
│   P(outcomes | actions) joint distribution                Persona    │
│                                                           │           │
├───────────────────────────────────────────────────────────┼───────────┤
│                       TRAINING                            │           │
│                                                           ▼           │
│   PPO-Lite (2×128 MLP) ── Gymnasium wrapper ──► Trained Actor-Critic │
│   50k steps, GAE, clipped objective            models/user_{id}/     │
│                                                           │           │
├───────────────────────────────────────────────────────────┼───────────┤
│                       INFERENCE                           │           │
│                                                           ▼           │
│   NN forward pass ──► optimal (sleep, exercise, nutrition) action    │
│        │                                                             │
│        ▼                                                             │
│   GPT-4o-mini synthesis ──► natural-language coaching + expected     │
│                              delta predictions + healthspan impact   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                       EVALUATION (next day)                          │
│                                                                      │
│   Expected Δ vs Actual Δ ──► fidelity_score (0.0–1.0)              │
│   Per-input compliance tracking (sleep / exercise / nutrition)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system architecture, data flow diagrams, module responsibilities, and refactoring roadmap.

---

## Project Structure

```
Wellness-Outcome/
│
├── app.py                          # FastAPI server — monolithic entry point (~40 endpoints)
├── inference.py                    # OpenEnv hackathon evaluation entrypoint (LLM agent)
├── openenv.yaml                    # OpenEnv task declarations
├── Dockerfile                      # 2-stage build: Node.js (webapp) → Python (server)
│
├── wellness_env/                   # OpenEnv-compliant Gymnasium environment
│   ├── env.py                      #   WellnessEnv — reset/step/state/grade interface
│   ├── models.py                   #   Pydantic models (Action, Biomarkers, StepResult, etc.)
│   ├── simulator.py                #   Rule-based physiological response model
│   ├── distribution_simulator.py   #   Gaussian copula alternative simulator
│   ├── graders.py                  #   Task graders (0.0–1.0 scoring)
│   ├── payoff.py                   #   Outcome-based reward function
│   └── personas.py                 #   4 persona archetypes with hidden ResponseModels
│
├── backend/                        # Production backend services
│   ├── database.py                 #   SQLAlchemy ORM, CRUD, schema migrations
│   ├── inference_service.py        #   Hybrid NN+LLM recommendation engine (4 fidelity levels)
│   ├── distribution_calibration.py #   Gaussian copula calibration from wearable history
│   ├── calibration.py              #   ⚠️ DUPLICATE — older OLS calibration (can delete)
│   ├── eval_service.py             #   Recommendation evaluation (fidelity + compliance)
│   ├── evals.py                    #   ⚠️ DUPLICATE — simpler eval (can delete)
│   ├── garmin_service.py           #   Garmin Connect API integration
│   ├── terra_service.py            #   Terra API for Apple Watch / Fitbit
│   ├── upload_service.py           #   Apple Health XML / CSV / JSON import
│   ├── llm_nutrition.py            #   GPT-4o-mini food log → calorie/macro parser
│   ├── persist.py                  #   HuggingFace Space persistence (DB + models)
│   ├── guardian.py                 #   ⚠️ DEAD CODE — process supervisor (unused in Docker)
│   └── calibrated_persona.json     #   Example calibrated persona output
│
├── rl_training/                    # PPO reinforcement learning
│   ├── ppo_lite.py                 #   ActorCritic network + PPO algorithm
│   ├── env_wrapper.py              #   Gymnasium wrapper (150 discrete actions → 19-float obs)
│   └── train.py                    #   Training loop (50k steps default)
│
├── webapp/                         # Next.js 16 frontend (static export → served by FastAPI)
│   └── src/
│       ├── app/                    #   Root page + layout + global CSS (dark theme)
│       └── components/
│           ├── GarminDashboard.tsx  #   Main dashboard (~1200 lines, multi-user, charts)
│           ├── AdminPanel.tsx       #   Admin user overview
│           ├── ManualLogForm.tsx    #   Food/weight/note entry with LLM parsing
│           └── UserManual.tsx       #   In-app help
│
├── models/                         # Trained model artifacts
│   └── ppo_wellness_lite.pt        #   Pre-trained baseline PPO model
│
├── tests/                          # pytest test suite
│   ├── test_env.py                 #   Environment episode tests
│   ├── test_graders.py             #   Grader score range / determinism
│   ├── test_payoff.py              #   Reward computation tests
│   ├── test_simulator.py           #   Biomarker delta direction tests
│   ├── test_distribution_calibration.py  # Copula fit/conditional tests
│   ├── test_distribution_simulator.py    # Distribution simulator bounds
│   ├── test_database_persistence.py      # DB CRUD integration
│   └── test_llm_nutrition.py       #   Live LLM parse test (needs API key)
│
├── static/                         # ⚠️ Built webapp output (can regenerate from webapp/)
├── hf_deploy/                      # ⚠️ DUPLICATE — deployment copy of entire project
├── hf_deploy 2/                    # ⚠️ DUPLICATE — partial deployment copy
├── scratch/                        # ⚠️ Experimental scripts (not used in production)
├── demo.py                         # ⚠️ Standalone chart generator (not used in production)
├── deploy_copy.py                  # ⚠️ Manual copy script (replaced by Docker)
├── hf_upload.py                    # ⚠️ Manual HF upload script
└── demo_output/                    # ⚠️ Generated demo charts
```

Items marked with ⚠️ are candidates for cleanup — see [Refactoring Notes](#refactoring-notes).

---

## OpenEnv Interface

The environment conforms to the [OpenEnv specification](https://github.com/OpenEnvs/openenv-core):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Initialize episode with `task_name`, returns `EnvState` |
| `/step` | POST | Submit `Action`, returns `StepResult` with reward |
| `/state` | GET | Current observation + task info |
| `/grade` | GET | Final 0.0–1.0 score from task-specific grader |
| `/health` | GET | Liveness check |

### Action Space (150 combinations)

| Dimension | Values |
|-----------|--------|
| **Sleep** | `very_short` (4h), `short` (5.5h), `moderate` (7h), `long` (8.5h), `very_long` (10h) |
| **Exercise** | `none`, `light_cardio`, `moderate_cardio`, `intense_cardio`, `strength`, `hiit` |
| **Nutrition** | `poor`, `below_average`, `moderate`, `good`, `excellent` |

### Observation Space (8 biomarkers)

| Biomarker | Range | Unit |
|-----------|-------|------|
| `resting_hr` | 40–100 | bpm |
| `hrv` | 10–120 | ms |
| `vo2_max` | 20–60 | mL/kg/min |
| `body_fat_pct` | 5–50 | % |
| `lean_mass_kg` | 30–80 | kg |
| `sleep_efficiency` | 0.5–1.0 | ratio |
| `cortisol_proxy` | 0.2–1.0 | normalized |
| `energy_level` | 0.1–1.0 | normalized |

### Tasks

| Task | Days | Persona | Difficulty | Primary Goal |
|------|------|---------|------------|--------------|
| `single_goal` | 14 | Athletic | Easy | Improve HRV |
| `multi_outcome` | 30 | Stressed | Medium | Balance cortisol + sleep + energy |
| `resistant_adaptation` | 30 | Weight Loss | Hard | Body composition change |

### Reward Function

$$R_t = 50 + \sum_{i=1}^{8} \text{normalize}(\Delta_i) \times w_{i,\text{goal}} \times 100$$

Clamped to [0, 100]. Blended: 70% biomarker deltas + 30% absolute state quality.

---

## Production Coaching System

Beyond the hackathon environment, the system includes a full production coaching pipeline:

### 4-Level Fidelity Ladder

| Level | Name | Trigger | Method |
|-------|------|---------|--------|
| 0 | Generic | No data | LLM autonomous coaching |
| 1 | Basic | <15 syncs | LLM with history context |
| 2 | Calibrated | 15+ syncs + approved persona | NN forward + LLM synthesis |
| 3 | AI-Optimized | Trained NN model exists | NN forward + LLM synthesis |

### Causal Lag Model

Behavioral inputs from day *T* are matched with biological outcomes from day *T+1*. This correctly models delayed physiological response — yesterday's sleep and exercise affect today's HRV and recovery.

### Wearable Integration

| Source | Method | Status |
|--------|--------|--------|
| Garmin Connect | Direct API scraping (garminconnect lib) | Active |
| Apple Watch | Terra API (OAuth + webhooks) | Active |
| Fitbit | Terra API | Supported |
| Apple Health Export | XML/CSV/JSON file upload | Active |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+ (for webapp build)
- OpenAI API key (for LLM features)

### Local Development

```bash
# Clone
git clone https://github.com/raviasha/Wellness-Outcome.git
cd Wellness-Outcome

# Python deps
pip install -r requirements.txt

# Build webapp (optional — pre-built static/ exists)
cd webapp && npm install && npm run build && cd ..

# Set environment
export OPENAI_API_KEY=sk-...

# Run
python app.py
# → http://localhost:7860
```

### Run Tests

```bash
pytest tests/ -v
```

### Run OpenEnv Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_...
python inference.py
```

---

## Environment Variables

### Required for Production

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini (nutrition parser, hybrid inference, evals) |

### Optional

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string (Neon.tech) | SQLite `wellness.db` |
| `GARMIN_EMAIL` | Garmin Connect email for auto-sync | — |
| `GARMIN_PASSWORD` | Garmin Connect password | — |
| `TERRA_API_KEY` | Terra API key for Apple Watch / Fitbit | — |
| `TERRA_DEV_ID` | Terra developer ID | — |
| `TERRA_WEBHOOK_SECRET` | Terra webhook signing secret | — |
| `FERNET_KEY` | Encryption key for stored credentials | Auto-generated |
| `HF_TOKEN` | HuggingFace token (for persistence + inference) | — |
| `SIMULATE` | Enable mock data mode (`true`/`false`) | `false` |

### OpenEnv Evaluation

| Variable | Purpose | Default |
|----------|---------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `SEED` | Reproducibility seed | `42` |

---

## Deployment

### Docker (HuggingFace Spaces)

```bash
docker build -t wellness-outcome .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... wellness-outcome
```

The Dockerfile uses a 2-stage build:
1. **Node.js 20** — builds the Next.js webapp to static HTML (`out/`)
2. **Python 3.11** — copies static output + runs FastAPI on port 7860

Container persistence is handled by `backend/persist.py`, which uploads `wellness.db` and trained models back to the HuggingFace Space git repo on each persist cycle.

---

## Tests

| Test File | What It Covers |
|-----------|----------------|
| `test_env.py` | Environment lifecycle, model validation, episode completion |
| `test_graders.py` | Score ranges [0,1], determinism, good > bad ordering |
| `test_payoff.py` | Goal weights, delta normalization, reward computation |
| `test_simulator.py` | Biomarker directional correctness, overtraining detection |
| `test_distribution_calibration.py` | Copula fit dimensions, PSD, conditional recovery |
| `test_distribution_simulator.py` | Bounds checking, full episode completion |
| `test_database_persistence.py` | DB CRUD integration tests |
| `test_llm_nutrition.py` | Live LLM food parsing (requires `OPENAI_API_KEY`) |

**Missing coverage:** API endpoints (`app.py`), `inference_service.py`, `eval_service.py`, `terra_service.py`, `upload_service.py`, webapp components.

---

## Refactoring Notes

### Files to Delete

| File/Directory | Reason |
|----------------|--------|
| `hf_deploy/` | Full duplicate of root project (deployment artifact) |
| `hf_deploy 2/` | Partial duplicate of root project |
| `backend/guardian.py` | Dead code — process supervisor not used in Docker CMD |
| `backend/evals.py` | Superseded by `backend/eval_service.py` |
| `backend/calibration.py` | Superseded by `backend/distribution_calibration.py` |
| `demo.py` | Standalone chart generator, not part of the app |
| `demo_output/` | Generated demo charts |
| `deploy_copy.py` | Manual script, replaced by CI/Docker |
| `hf_upload.py` | Manual HF upload script |
| `scratch/` | Experimental scripts (bootstrap_data, test_regression, verify_brain) |
| `payload.json`, `payload2.json`, `format2.json`, `formatted_payload.json` | Sample Garmin API responses — dev artifacts |
| `export_cda.xml` | Sample CDA export — dev artifact |
| `static/_next 2/`, `static/_next 3/` | Stale build artifacts |
| `guardian.log`, `guardian_crash.log`, `server.log` | Log files (should be gitignored) |

### Code to Refactor

| Issue | Location | Suggestion |
|-------|----------|------------|
| **Monolithic server** | `app.py` (~1115 lines) | Split into FastAPI routers: `routers/openenv.py`, `routers/users.py`, `routers/wearable.py`, `routers/nutrition.py`, `routers/training.py`, `routers/admin.py` |
| **Hardcoded Fernet fallback** | `backend/database.py` | Remove `FALLBACK_KEY`, require `FERNET_KEY` env var or generate + persist |
| **No API authentication** | `app.py` | `X-User-Id` header is trusted without verification — add auth middleware |
| **In-memory sync backoff** | `app.py` `_sync_backoff` | Lost on restart — persist to DB or use proper rate limiter |
| **GarminDashboard mega-component** | `webapp/src/components/GarminDashboard.tsx` (~1200 lines) | Split into smaller components: `BiometricCharts`, `RecommendationPanel`, `SyncControls`, `HistoryTable` |
| **LLM in DB layer** | `backend/database.py` `add_manual_log()` | Food deduplication via LLM belongs in a service layer, not the DB module |
| **Static build artifacts in git** | `static/` | Should be gitignored and only generated during Docker build |

### .gitignore Additions Needed

```
# Log files
*.log

# Dev artifacts
payload*.json
format*.json
formatted_payload.json
export_cda.xml
```

---

## License

[MIT](LICENSE)
