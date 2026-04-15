# Round 1 Submission Checklist

## What you need to submit
- A **Hugging Face Space URL** (e.g. `https://raviasha-wellness-outcome.hf.space`)
- That Space must respond to `POST /reset` with HTTP 200

---

## Step 1 — FastAPI server + Dockerfile

`app.py` is already in this repo. It exposes the environment over HTTP so the
validator can call `POST /reset`. The Dockerfile runs `app.py` on port 7860.
`fastapi` and `uvicorn` are already in `requirements.txt`.

---

## Step 2 — Test Docker build locally

```bash
cd /path/to/Wellness-Outcome
docker build -t wellness-outcome .
docker run -p 7860:7860 wellness-outcome
# In another terminal:
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name": "single_goal"}'
# Should return JSON with observation
```

---

## Step 3 — Create Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Set:
   - **Owner**: raviasha
   - **Space name**: Wellness-Outcome (or anything)
   - **SDK**: Docker
   - **Visibility**: Public
3. Click **Create Space**

---

## Step 4 — Push repo to the HF Space

```bash
cd /path/to/Wellness-Outcome

# Add HF Space as a remote
git remote add space https://huggingface.co/spaces/raviasha/<your-space-name>

# Push
git push space main
```

Or link your GitHub repo directly in the Space settings under "Files → Connect a repository".

---

## Step 5 — Set Space secrets

In the HF Space → **Settings → Repository secrets**, add:

| Name | Value |
|------|-------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Your HF token (from https://huggingface.co/settings/tokens) |

---

## Step 6 — Verify Space is live

Wait for the Space to build (1–3 min), then:

```bash
curl -X POST https://raviasha-<space-name>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "single_goal"}'
# Must return HTTP 200
```

---

## Step 7 — Run the pre-submission validator

```bash
pip install openenv-core

# Download and run the validator script
bash validate-submission.sh https://raviasha-<space-name>.hf.space /path/to/Wellness-Outcome
```

All 3 checks must pass:
- [x] HF Space responds to /reset
- [x] docker build succeeds
- [x] openenv validate passes

---

## Step 8 — Verify inference.py runs end-to-end

```bash
cd /path/to/Wellness-Outcome
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

Expected output format:
```
[START] task=single_goal env=wellness-outcome model=Qwen/Qwen2.5-72B-Instruct ...
[STEP] step=1 action={...} reward=55.12 done=false error=null ...
...
[END] success=true steps=14 score=0.6800 rewards=55.12,...

[START] task=multi_outcome ...
...
[END] success=true steps=30 score=0.6200 rewards=...

[START] task=resistant_adaptation ...
...
[END] success=true steps=30 score=0.4800 rewards=...
```

Must complete in under 20 minutes on 2 vCPU / 8 GB RAM.

---

## Step 9 — Submit

Submit your HF Space URL to the hackathon portal.

---

## Disqualification checks (must all pass)chatgp

- [ ] HF Space deploys and responds to `POST /reset`
- [ ] `docker build` succeeds
- [ ] `openenv validate` passes
- [ ] `inference.py` runs without error and produces scores
- [ ] 3 tasks with graders, scores in [0.0, 1.0]
- [ ] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` read from env vars
- [ ] OpenAI client used for all LLM calls
- [ ] `inference.py` at root of project
- [ ] `[START]`, `[STEP]`, `[END]` stdout format correct

---

## Environment variable reference

| Variable | Where set | Purpose |
|----------|-----------|---------|
| `API_BASE_URL` | HF Space secret | LLM API endpoint |
| `MODEL_NAME` | HF Space secret | Model identifier |
| `HF_TOKEN` | HF Space secret | Auth for HF router |
| `OPENAI_API_KEY` | Optional local | Falls back to HF_TOKEN |
| `SEED` | Optional | Reproducibility (default: 42) |
