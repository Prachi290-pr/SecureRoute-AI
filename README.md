---
title: SecureRoute AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# SecureRoute-AI

SecureRoute-AI is a text-based OpenEnv environment for enterprise support-ticket triage. An agent receives a raw ticket, redacts required PII, and routes the ticket to the right team.

## Overview

This project is designed for deterministic evaluation and lightweight deployment.

- Observation: support-ticket text
- Action: redacted ticket text plus routing decision
- Reward: normalized score from `0.0` to `1.0`

## Scoring

The environment uses a simple additive score:

- `+0.3` for correct routing
- `+0.7` for exact redaction
- `0.0` if sensitive data leaks in the submitted text

Each episode is single-step, so a single `step()` call produces the final reward.

## Benchmark Tasks

- Easy: ticket `1`, a normal IT support request
- Medium: ticket `3`, billing with a credit-card redaction requirement
- Hard: ticket `10`, phishing plus SSN redaction for Security

## Repository Layout

- `openenv.yaml`: OpenEnv metadata
- `models.py`: Pydantic models and routing enum
- `environment.py`: `reset()`, `state()`, and `step()` implementation
- `graders.py`: deterministic scoring helpers
- `inference.py`: OpenAI-compatible inference runner with `[START]`, `[STEP]`, `[END]` logs
- `app.py`: FastAPI HTTP wrapper for `/reset`, `/state`, and `/step`
- `server/app.py`: packaged server entry point for deployment checks
- `deploy_space.py`: Hugging Face Space deployment helper
- `tickets.json`: benchmark tickets
- `Dockerfile`: container runtime for Hugging Face Spaces
- `pyproject.toml`: project metadata and script entry point
- `uv.lock`: resolved dependency lockfile

## Runtime Variables

`inference.py` reads the following environment variables:

- `API_BASE_URL` for the OpenAI-compatible client endpoint
- `MODEL_NAME` for the model identifier
- `HF_TOKEN` for authentication
- `LOCAL_IMAGE_NAME` as an optional harness hint

## HTTP API

Run locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /` returns the available endpoints
- `GET /health` returns a basic status payload
- `POST /reset` accepts no body, `{}`, or `{"ticket_id": 3}`
- `GET /state` returns the current observation
- `POST /step` accepts `redacted_text` and `routing`

## Local Commands

```bash
pip install -r requirements.txt
python inference.py
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

The deployment helper uploads the repository and can also set Space secrets from available environment variables: `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`.

## Collaborators

- Chintamani Joshi @Chintamani26
- Prachi Sawant @Prachi290-pr
