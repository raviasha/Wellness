# Wellness-Outcome System Specifications

This document outlines the reverse-engineered specifications for the Wellness-Outcome platform based on the current codebase implementation.

## 1. System Overview
Wellness-Outcome is a personalized health coaching platform that bridges the gap between wearable data and actionable lifestyle recommendations. The system ingests data from Garmin and Terra, learns individual physiological responses using a "fidelity ladder" (Rules → Copula → ML → Neural Network), and provides daily, goal-oriented coaching.

## 2. Core Architecture & Modules

The platform is a monolithic repository containing a FastAPI backend, a core reinforcement learning / simulation environment, and a statically exported Next.js frontend.

### 2.1 Backend (`backend/`)
- **`database.py`**: Defines the SQLAlchemy schema (SQLite/PostgreSQL) and data persistence logic. Key entities include `User`, `UserProfile`, `WearableSync`, `ManualLog`, and `Recommendation`.
- **`inference_service.py`**: Orchestrates the coaching logic across the 4-level fidelity ladder. Evaluates which predictive model to use based on the user's data maturity.
- **`garmin_service.py`**: Manages Garmin API integration, credential encryption, hybrid sync strategies, and smart data merging to prevent partial data overwrites.
- **`terra_service.py`**: Processes real-time push data from Terra webhooks, standardizing the payload before saving.
- **`goal_interpreter.py`**: Uses an LLM to interpret free-text user goals into structured outcome weights (`GoalProfile`).
- **`distribution_calibration.py`**: Implements Gaussian Copula calibration using Ledoit-Wolf shrinkage to regularize covariance matrices for predicting biomarker shifts with small sample sizes.
- **`outcome_models.py`**: Ridge regression models trained individually per outcome (e.g., resting HR, HRV) for Level 2.5 maturity.
- **`eval_service.py` / `eval_models_service.py`**: Evaluates historical recommendations, computing fidelity scores by comparing predicted biomarker deltas against actual observed deltas.
- **`feature_engineering.py`**: Constructs the $X$ (actions/lags) and $Y$ (outcome deltas) matrices aligning time $T$ features with $T+1$ physiological outcomes.
- **`maturity_config.py`**: Manages user progression through the fidelity tiers based on the number of days of available paired data.
- **`persist.py`**: Handles synchronization of the local SQLite database and model artifacts with the upstream Hugging Face Space repository.

### 2.2 Core Domain (`wellness_env/`)
- **`models.py`**: Pydantic definitions for core domain constructs (e.g., `Action`, `Biomarkers`, `BiomarkerDeltas`, `RewardBreakdown`).
- **`env.py`**: The `WellnessEnv` class implementing a Gymnasium-compatible reinforcement learning environment.
- **`payoff.py`**: Computes the reward function mapping biomarker deltas and goals to a [0, 100] normalized score.
- **`simulator.py`**: A physiological rule-based simulator used during early fidelity levels and to pre-train/augment the Neural Network.
- **`personas.py`**: Hidden response models defining unique baseline physiological traits and sensitivities for various user archetypes.

### 2.3 Reinforcement Learning (`rl_training/`)
- **`ppo_lite.py`**: A custom implementation of Proximal Policy Optimization (PPO) using a multi-headed Actor-Critic architecture to support multi-discrete action spaces.
- **`train.py`**: The training loop orchestrating environment setup, PPO training, and model checkpointing.

### 2.4 Application Entrypoint
- **`app.py`**: A 1,600+ line FastAPI monolith that defines all API routes, background training tasks, webhook handlers, and the SPA (Single Page Application) static file server.

### 2.5 Frontend (`webapp/`)
- A React-based application built with Next.js (statically exported). Key components include `GarminDashboard.tsx` for viewing syncs and recommendations, and `OutcomeModelEvals.tsx` for model transparency.

---

## 3. Data Flows

### 3.1 Data Ingestion Flow
1. **Source**: Data arrives either via proactive polling to Garmin (`/api/sync`) or via passive Terra webhooks (`/api/terra/webhook`).
2. **Standardization**: Both pathways normalize data into a consistent internal dictionary representing daily actions and biomarkers.
3. **Smart Merging**: `database.save_wearable_sync` executes a merge operation to ensure partial updates (e.g., mid-day syncs) do not overwrite more comprehensive daily aggregates.

### 3.2 Recommendation Generation Flow
1. **Request**: The frontend requests `/api/recommendations`.
2. **Context Assembly**: The system retrieves the latest `WearableSync` and the user's active `GoalProfile`.
3. **Fidelity Selection**: The system checks the user's active tier (via `maturity_config`):
   - *Level 0/1*: Uses baseline persona / simulator rules.
   - *Level 2*: Uses calibrated Gaussian Copula.
   - *Level 2.5*: Uses Ridge Regression models.
   - *Level 3*: Uses the trained PPO Neural Network.
4. **Generation**: The chosen model outputs expected biomarker deltas based on proposed actions.
5. **Synthesis**: The expected deltas and actions are passed to an LLM (`_generate_llm_recommendation`) to produce natural language coaching.

### 3.3 Training & Calibration Flow
1. **Pairing**: `feature_engineering.py` extracts actions at time $T$ and biomarker changes at time $T+1$.
2. **Fitting**:
   - For Copula, Ledoit-Wolf shrinkage is applied to the covariance matrix.
   - For ML, Ridge Regression is fitted per outcome.
   - For NN, the PPO agent interacts with the `WellnessEnv` (which wraps either the simulator or the copula models) to maximize the reward defined in `payoff.py`.

---

## 4. API Endpoints

### 4.1 Authentication & Profile
- `POST /api/auth/garmin`: Connect a Garmin account (encrypts credentials).
- `POST /api/terra/connect`: Connect Terra provider.
- `POST /api/profile`: Update user profile details.

### 4.2 Data Synchronization
- `POST /api/sync`: Manually trigger a Garmin data sync for up to 15 past days.
- `POST /api/terra/webhook`: Receive real-time data payloads from Terra.
- `POST /api/logs/manual`: Submit manual logs (e.g., nutrition, hydration).

### 4.3 Goals & Periodization
- `POST /api/user/goal`: Set a free-text goal and optional target date (processed by LLM).
- `GET /api/user/goal`: Retrieve the current active goal.
- `DELETE /api/user/goal`: Revert to standard preset goals.

### 4.4 Recommendations & Evaluation
- `GET /api/recommendations`: Generate daily coaching recommendations.
- `GET /api/persona/evals`: Retrieve historical evaluation fidelity scores.
- `GET /api/evals/models`: Get outcome model statistics (e.g., $R^2$ scores).
- `GET /api/evals/inference-comparison`: Compare predictions from the primary model vs. alternative models over the last 7 days.

### 4.5 Maturity Ladder Management
- `GET /api/maturity/status`: Check current maturity tier and required data gates.
- `POST /api/maturity/transition`: Advance to the next tier if requirements are met.
- `POST /api/maturity/revert`: Fall back to a lower tier.
- `POST /api/calibrate`: Run Gaussian Copula calibration manually.
- `POST /api/train`: Trigger background training for the PPO Neural Network.

### 4.6 Administration
- `POST /api/persist`: Push SQLite database and user model artifacts to the Hugging Face repository.
- `GET /api/admin/data`: Export comprehensive tables (Users, Profiles, Syncs, Logs) for the admin dashboard.
- `POST /api/admin/backfill-raw` / `backfill-activities`: Administrative data repair utilities.

---

## 5. Security & Infrastructure Notes

- **Credential Storage**: Third-party credentials (like Garmin passwords) are symmetrically encrypted using Fernet. The `FERNET_KEY` is sourced from environment variables, falling back to a hardcoded key if absent.
- **Authentication**: Current implementation trusts the `X-User-Id` HTTP header. For production deployment, this must be replaced with session tokens or JWTs.
- **Persistence**: Given the ephemeral nature of Hugging Face Spaces, state modifications are periodically committed back to the Git repository via background tasks in `persist.py`.
- **Frontend Hosting**: The compiled Next.js SPA is served via FastAPI's catch-all route (`/{full_path:path}`) at the bottom of `app.py`.
