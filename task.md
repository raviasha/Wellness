# Wellness App Execution Tasks (Granular)

To avoid context limits, these tasks are isolated and can be built sequentially. When starting a new development session, simply ask the model to look at this checklist and start the next unfinished item.

## Phase 1: Data Collection interface
- `[x]` **Task 1.1: Web App Initialization**
- `[x]` **Task 1.2: Garmin Backend Service**
- `[x]` **Task 1.3: Web Dashboard UI**
- `[x]` **Task 1.4: Manual Logging UI**

## Phase 2: Simulator Calibration
- `[x]` **Task 2.1: Persona Refactor**
- `[x]` **Task 2.2: Physics Calibration**

## Phase 3: Persistence & Intelligence (Foundation)
- `[x]` **Task 4.1: SQLite Database Setup**
- `[x]` **Task 4.2: AI Nutrition Parser**
  - Implement `backend/llm_nutrition.py` using GPT-4o-mini.
  - Convert natural language food logs (e.g., "I ate a burrito") into structured `NutritionType` and macro estimates.
- `[x]` **Task 4.3: Real-Time UI Reflection**
  - Update `ManualLogForm.tsx` to display AI-extracted nutrition data.
  - Ensure the Dashboard charts pull from the live SQLite history.

## Phase 4: Neural Network Training & Inference (The Brain)
- `[x]` **Task 3.1: Gymnasium Wrapper**
  - Create the `rl_training/env_wrapper.py` adapting OpenEnv to Stable Baselines3 continuous state vectors.
  - Map discrete action spaces.
- `[ ]` **Task 3.2: Model Training**
  - Write `train.py`.
  - Execute training loop until convergence.
- `[ ]` **Task 3.3: Hybrid Inference Layer (RL + LLM)**
  - Link the trained `.zip` model file to the FastAPI backend to generate optimal numeric actions.
  - Add an LLM processing step that takes the RL action and raw Garmin state to synthesize a human-readable recommendation.
  - Display the final LLM-guided recommendations on the Next.js Dashboard.

## Phase 5: Universal Cloud Deployment
- `[ ]` **Task 5.1: Production Dockerization**
- `[ ]` **Task 5.2: Frontend Vercel Deploy**
- `[ ]` **Task 5.3: Automated Auto-Sync**

## Phase 6: Local-to-Cloud NN Bridge
- `[ ]` **Task 6.1: Cloud Inference Service**
