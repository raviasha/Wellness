# Wellness-Outcome Codebase Architecture Document

This document serves as a persistent reference of the `Wellness-Outcome` codebase, detailing how the simulation works and the bottlenecks expected when transitioning to a real-world neural network deployment.

## System Architecture & Data Flow

The environment currently mimics a standard reinforcement learning `gym` interface but relies on complex stochastic biological models under the hood.

### 1. The Environment Engine (`wellness_env/env.py`)
This is the core state machine.
- `reset(task_name)`: Mounts a specific persona (`athletic_performance`, `stress_management`, `weight_loss`) and gives them 14-30 days to improve.
- `step(action)`: Advances the simulation by exactly one day. It takes an `Action` Pydantic model. 

### 2. The Hidden Biological Mechanics
During a `step()`, the environment sequentially executes several hidden factors:
1. **Compliance (`personas.py`)**: The user's persona determines if they actually followed the agent's action. (e.g. they might eat junk instead of a balanced meal).
2. **Life Events (`simulator.py`)**: Injects realistic stochasticity (like sudden sickness or poor unexpected sleep).
3. **Response Model**: Converts the `actual_action` into physiological **deltas** (changes to the body). Every persona reacts differently to the same action.

### 3. The 8 Biomarkers
All state observations and rewards revolve around these 8 metrics:
1. `resting_hr`
2. `hrv` 
3. `vo2_max`
4. `body_fat_pct`
5. `lean_mass_kg`
6. `sleep_efficiency`
7. `cortisol_proxy`
8. `energy_level`

### 4. Reward Calculation (`payoff.py`)
Rewards are not stationary. They are **Goal-Weighted**.
- A `weight_loss` persona heavily weights improvements to body fat %.
- An `athletic_performance` persona heavily weights VO2 max and HRV.
The agent must optimize for these specific payoff functions.

### 5. Agents (`inference.py`)
The current inference pipeline is prompt-based:
- **LLM Agent**: Sends system prompts containing the 7-day history and current state to GPT-4o-mini to return a JSON containing `sleep`, `exercise`, and `nutrition`.
- **Fallback Agent**: Simple hardcoded rules (e.g., "If energy is low -> Do Yoga and Sleep Optimal High").

---

## Technical Considerations for Phase 2 Deployment

To migrate to an LSTM/PPO neural network that runs a real-world web/mobile app backend, we have to tackle the following structural differences:

### Simulation vs Asynchronous Reality
- **Currently**: The agent runs a tight continuous loop over arrays.
- **Future State**: The app is disconnected. A user might miss logging a day, or log three days later. The environment's physical response model (`simulator.py`) gets replaced by the user's actual body. We only use `simulator.py` to **train** the network, while in deployment, we just query the NN directly.

### Action / State Encodings
The NN can't natively read JSON strings like `"moderate_cardio"`.
We'll need to write serializers/wrappers that map the discrete string spaces to numerical tensors (e.g., a MultiDiscrete space of sizes 5, 6, and 5 for Sleep/Exercise/Nutrition categories).
