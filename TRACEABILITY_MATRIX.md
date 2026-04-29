# Requirements Traceability Matrix

This document maps the functional and non-functional requirements from the `REQUIREMENTS.md` document to the specific test cases and modules within the codebase. 

*Note: Some system-level, UI, and external integration requirements rely on end-to-end (E2E) testing, manual verification, or are verified implicitly through architectural tests rather than unit tests.*

## Traceability Matrix

| Req ID | Requirement Name | Associated Test Cases / Modules | Coverage Status |
| :--- | :--- | :--- | :--- |
| **REQ-1** | Garmin Synchronization | *Requires E2E testing (External API)* | Manual / Uncovered |
| **REQ-2** | Multi-Wearable Support (Terra) | *Requires E2E testing (External API)* | Manual / Uncovered |
| **REQ-3** | Smart Data Merging | `test_database_persistence` | Partially Covered |
| **REQ-4** | Manual Logging | `test_nutrition_parser` | Covered |
| **REQ-5** | Free-Text Goals | *Planned/Pending LLM tests* | Uncovered |
| **REQ-6** | Goal Interpretation | *Planned/Pending LLM tests* | Uncovered |
| **REQ-7** | Target Dates & Periodization | *Planned/Pending LLM tests* | Uncovered |
| **REQ-8** | Preset Goals | `test_goal_enum`, `test_persona_goals`, `test_all_goals_have_weights` | Covered |
| **REQ-9** | Daily Action Plan | `test_action_valid`, `test_action_from_string` | Covered |
| **REQ-10**| Natural Language Explanations | *Generative AI Module (evals based)* | Uncovered |
| **REQ-11**| Outcome Projections | `test_compute_returns_deltas`, `test_all_deltas_within_physiological_bounds`, `test_reward_breakdown_fields` | Covered |
| **REQ-12**| Sport-Specific Advice | *Generative AI Module (evals based)* | Uncovered |
| **REQ-13**| Tier 1 (Rules-Based) | `test_simulator.py` (e.g., `test_good_sleep_improves_hrv`, `test_overtraining_hurts_hrv`) | Covered |
| **REQ-14**| Tier 2 (Statistical Calibration)| `test_distribution_calibration.py` (e.g., `test_returns_joint_distribution`, `test_conditional_mean_for_known_linear_relationship`, `test_shrinkage_in_range`) | Covered |
| **REQ-15**| Tier 3 (Machine Learning) | *Requires ML validation suite* | Manual / Evals |
| **REQ-16**| Tier 4 (Reinforcement Learning) | `test_env.py` (e.g., `test_full_episode_runs_without_error`, `test_episode_completes`, `test_reward_total_in_range`) | Covered |
| **REQ-17**| Fidelity Scoring | `test_graders.py` (e.g., `test_grade_matches_grader`, `test_grade_returns_valid_score`) | Covered |
| **REQ-18**| Compliance Tracking | `test_apply_compliance_deterministic_when_complied`, `test_apply_compliance_never_when_zero` | Covered |
| **REQ-19**| Model Transparency | `test_distribution_calibration.py` / *UI E2E* | Partially Covered |
| **REQ-20**| Credential Protection | *Security / Penetration Testing* | Uncovered |
| **REQ-21**| Data Isolation | `test_database_persistence` | Partially Covered |
| **REQ-22**| Secure Integrations | *Security Testing (HMAC Verification)* | Uncovered |
| **REQ-23**| Dashboard Interface | *UI E2E Testing (Playwright/Cypress)* | Uncovered |
| **REQ-24**| Admin Visibility | *UI E2E Testing* | Uncovered |
| **REQ-25**| Cloud Deployability | *CI/CD & Docker Build Checks* | Implicit |
| **REQ-26**| Resilience | `test_database_persistence` / *Chaos Testing* | Partially Covered |

## Summary of Coverage
- **Core Domain & Simulation (Tier 1 & 4)**: Highly covered with extensive unit tests (`test_simulator.py`, `test_env.py`, `test_payoff.py`).
- **Statistical Models (Tier 2)**: Highly covered, particularly mathematical constraints (`test_distribution_calibration.py`).
- **External Integrations & Generative AI**: Currently lack automated unit tests. These modules require either mocking (for APIs) or evaluation-based testing (for LLMs).
- **UI & Security**: Rely on manual testing and implicit architecture. E2E test suites (like Cypress or Playwright) should be introduced to cover Dashboard and Admin visibility.
