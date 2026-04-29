# Product Requirements Document

This document outlines the functional and non-functional requirements for the Wellness-Outcome platform. It describes *what* the application is supposed to do, rather than *how* it is implemented.

## 1. Product Vision
Wellness-Outcome is a personalized health coaching platform that bridges the gap between wearable data and actionable lifestyle recommendations. The primary goal is to provide users with a "Causal AI Coach" that learns their unique physiological responses over time and gives tailored daily advice to optimize their health, recovery, and performance.

## 2. User Personas
1. **The Data-Driven Athlete**: Prepares for specific events (e.g., marathons, tournaments) and needs periodized training and recovery plans based on their actual physiological state.
2. **The Health Enthusiast**: Wants to improve specific biomarkers (e.g., lower resting heart rate, increase HRV, improve sleep quality) but lacks the knowledge to connect daily habits to biological outcomes.
3. **The Busy Professional**: Wants simple, clear, and adaptive instructions on when to sleep and how much to exercise to manage stress and maximize energy (body battery).

---

## 3. Functional Requirements

### 3.1. Wearable Integrations & Data Ingestion
- **Garmin Synchronization**: The system must authenticate with Garmin accounts and ingest daily summaries, activities, and granular biomarker data (sleep, HRV, resting HR, stress, body battery, VO2 max).
- **Multi-Wearable Support (Terra)**: The system must be capable of receiving real-time webhook data from Terra to support users on non-Garmin devices (e.g., Apple Watch, Oura, Fitbit).
- **Smart Data Merging**: The system must handle partial, intra-day data syncs seamlessly, updating metrics as they become available without overwriting previously captured high-fidelity data.
- **Manual Logging**: Users must be able to manually input health metrics, such as nutrition or hydration, using natural language.

### 3.2. Goals & Periodization
- **Free-Text Goals**: Users must be able to set custom goals using natural language (e.g., "prepare for a pickleball tournament").
- **Goal Interpretation**: The system must automatically interpret free-text goals into specific physiological targets and exercise preferences.
- **Target Dates & Periodization**: Users must be able to set target dates for their goals. The system must adapt its recommendations based on the timeline (e.g., base-building months out, tapering days before the event).
- **Preset Goals**: The system must offer fallback preset goals (e.g., "Stress Management", "Cardiovascular Fitness", "Sleep Optimization").

### 3.3. Coaching & Recommendations
- **Daily Action Plan**: The system must provide a daily recommendation specifying target sleep duration, bedtime window, activity level, exercise type, and exercise duration.
- **Natural Language Explanations**: Recommendations must be accompanied by an easy-to-understand explanation of *why* the actions are recommended based on the user's current data.
- **Outcome Projections**: The system must display the expected impact of the recommended actions on the user's key biomarkers (e.g., "+5 ms HRV", "-2 bpm RHR").
- **Sport-Specific Advice**: For custom goals, the coaching must incorporate sport-specific training advice.

### 3.4. Progressive Personalization (Fidelity Ladder)
The system must adapt to the amount of data available for a user, progressively shifting from generic advice to highly personalized modeling.
- **Tier 1 (Rules-Based)**: For new users, the system must provide evidence-based, heuristic recommendations.
- **Tier 2 (Statistical Calibration)**: Once enough data is gathered (e.g., 15-30 days), the system must establish a baseline mapping of the user's habits to their outcomes.
- **Tier 3 (Machine Learning)**: The system must eventually predict individual biomarker shifts using machine learning models trained specifically on the user's historical data.
- **Tier 4 (Reinforcement Learning)**: At full maturity, the system must use AI to discover optimal long-term strategies, balancing short-term fatigue with long-term fitness gains.

### 3.5. Transparency & Evaluation
- **Fidelity Scoring**: The system must evaluate its own past recommendations by comparing what it predicted would happen to a user's biomarkers against what actually happened.
- **Compliance Tracking**: The system must track whether the user followed the daily recommendations and calculate a compliance score.
- **Model Transparency**: The user must be able to view which lifestyle factors (e.g., sleep debt, exercise duration) have the highest impact on their specific biomarkers.

---

## 4. Non-Functional Requirements

### 4.1. Privacy & Security
- **Credential Protection**: User credentials for third-party wearable integrations must be securely encrypted at rest.
- **Data Isolation**: A user must only ever have access to their own physiological data and recommendations.
- **Secure Integrations**: Webhooks from third-party aggregators must be verified via HMAC signatures.

### 4.2. Usability & Access
- **Dashboard Interface**: The application must provide a responsive, single-page web dashboard for users to view their data, goals, and daily recommendations.
- **Admin Visibility**: The system must provide an administrative view to monitor system health, user counts, and data synchronization metrics.

### 4.3. Infrastructure
- **Cloud Deployability**: The application must be capable of running continuously in a cloud environment (such as Hugging Face Spaces).
- **Resilience**: The system must gracefully handle wearable API rate limits, network timeouts, and missing data fields without crashing the coaching generation pipeline.
