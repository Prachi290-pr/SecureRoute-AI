from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from environment import SecureRouteEnv
from models import Action, Observation, Reward


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


app = FastAPI(title="SecureRoute-AI API", version="1.0.0")
env = SecureRouteEnv()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, list[str]]:
    return {"endpoints": ["/health", "/reset", "/state", "/step"]}


@app.post("/reset", response_model=Observation)
async def reset(request: Request, ticket_id: int | None = None) -> Observation:
    try:
        selected_ticket_id = ticket_id

        raw_body = await request.body()
        if raw_body:
            try:
                payload = await request.json()
            except Exception:
                payload = None

            if isinstance(payload, dict):
                body_ticket_id = payload.get("ticket_id")
                if isinstance(body_ticket_id, int) or body_ticket_id is None:
                    selected_ticket_id = body_ticket_id

        return env.reset(ticket_id=selected_ticket_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=Observation)
def state() -> Observation:
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
        return StepResponse(observation=observation, reward=reward, done=done, info=info)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
