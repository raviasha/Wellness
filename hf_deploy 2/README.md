---
title: Wellness Outcome Digital Twin
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Wellness Outcome: Autonomous AI Health Coach

This is a personalized digital twin application that uses Garmin health data and Reinforcement Learning to provide optimized coaching.

### Features
- **Garmin Sync**: Real-world biomarker tracking (HRV, RHR, Intensity Minutes).
- **Personalized Calibration**: Linear Regression engine derived from your history.
- **AI Brain**: Custom PPO-Lite Reinforcement Learning agent.
- **Hybrid Inference**: NN-optimized actions explained by GPT-4o-mini.

### Deployment Config (Hugging Face Secrets)
To use this space, you must set the following Secrets:
- `OPENAI_API_KEY`: Your OpenAI key.
- `DATABASE_URL`: Your Neon.tech PostgreSQL connection string.
- `GARMIN_EMAIL`: Your Garmin Connect email.
- `GARMIN_PASSWORD`: Your Garmin Connect password.

### Local Development
1. `pip install -r requirements.txt`
2. `python app.py`
3. Visit `http://localhost:7860`
