---
title: SecureRoute AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# SecureRoute-AI

SecureRoute-AI is a text-based OpenEnv environment for PII redaction and compliance routing on customer support tickets.

## What The Agent Must Do

1. Redact required PII spans with exactly `[REDACTED]`.
2. Route each ticket to one department:
   - `IT`
   - `BILLING`
   - `SECURITY`

## Environment Contract

- Observation: ticket text (`Observation.text`)
- Action: `redacted_text` + `routing` (`Action`)
- Reward: float in `[0.0, 1.0]` (`Reward.score`)

### Reward Logic

- `+0.3` for correct routing
- `+0.7` for exact expected redaction
- `0.0` hard fail if any sensitive span leaks in `redacted_text`

Each episode is single-step (`done=True` after one `step`).

## Tasks

- Easy: ticket `1` (normal IT ticket)
- Medium: ticket `3` (credit card redaction + BILLING)
- Hard: ticket `10` (phishing + SSN redaction + SECURITY)

## Project Files

- `openenv.yaml`: OpenEnv metadata
- `models.py`: Pydantic models and enums
- `environment.py`: `reset()`, `state()`, `step()`
- `graders.py`: deterministic task graders
- `inference.py`: OpenAI-client inference runner with strict logs
- `app.py`: FastAPI HTTP wrapper (`/reset`, `/state`, `/step`)
- `server/app.py`: packaged server entry point for deployment checks
- `deploy_space.py`: Hugging Face Space deployment helper
- `tickets.json`: benchmark tickets
- `Dockerfile`: container runtime for Hugging Face Spaces
- `pyproject.toml`: project metadata and script entry point
- `uv.lock`: dependency lockfile for `uv`

## Inference Environment Variables

`inference.py` uses:

- `API_BASE_URL` (default provided)
- `MODEL_NAME` (default provided)
- `HF_TOKEN` (no default; optional for offline deterministic fallback)
- `LOCAL_IMAGE_NAME` (optional; for harness compatibility)

## Logging Format

`inference.py` emits strict structured logs:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## HTTP API

Start locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /` returns available endpoints
- `GET /health`
- `POST /reset` with `{ "ticket_id": 3 }` or `{}`
- `GET /state`
- `POST /step` with:

```json
{
  "redacted_text": "...",
  "routing": "IT"
}
```

## Local Run

```bash
pip install -r requirements.txt
python inference.py
```

## Docker

```bash
docker build -t secureroute-ai .
docker run -p 8000:8000 secureroute-ai
```

## Deploy To Hugging Face Spaces

```bash
export HF_TOKEN=...
export HF_SPACE_ID=your-username/secure-route-ai
python deploy_space.py
```

Optional:

- `HF_SPACE_PRIVATE=true`

The deployment helper uploads the repo and can also set Space secrets from available environment variables (`HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`).
