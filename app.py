from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from environment import SecureRouteEnv
from models import Action, Observation, Reward


class ResetRequest(BaseModel):
    ticket_id: int | None = Field(default=None)


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
def reset(request: ResetRequest = ResetRequest()) -> Observation:
    try:
        return env.reset(ticket_id=request.ticket_id)
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
