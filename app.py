from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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
def root() -> HTMLResponse:
        return HTMLResponse(
                content="""
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SecureRoute-AI</title>
        <style>
            :root {
                --bg: #0b1220;
                --card: #111a2b;
                --text: #e5ecf4;
                --muted: #98a6ba;
                --accent: #3ddc97;
                --border: #25324a;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Segoe UI, sans-serif;
                color: var(--text);
                background: radial-gradient(circle at top right, #12305a 0%, var(--bg) 50%);
            }
            .wrap {
                max-width: 860px;
                margin: 48px auto;
                padding: 0 20px;
            }
            .card {
                background: linear-gradient(180deg, #121d30 0%, var(--card) 100%);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 28px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
            }
            h1 {
                margin: 0 0 8px;
                font-size: 30px;
                letter-spacing: 0.2px;
            }
            p {
                color: var(--muted);
                line-height: 1.6;
                margin: 0 0 18px;
            }
            ul {
                list-style: none;
                padding: 0;
                margin: 0;
                display: grid;
                gap: 10px;
            }
            li {
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 12px 14px;
                background: #0f1828;
            }
            code {
                color: var(--accent);
                font-size: 14px;
            }
            a {
                color: #8bd3ff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .btn {
                display: inline-block;
                margin-top: 12px;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 10px 12px;
                background: #12233a;
                color: #cde9ff;
                cursor: pointer;
            }
            .btn:hover {
                background: #17304f;
            }
            pre {
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 10px;
                background: #0f1828;
                color: #dbe8f5;
                white-space: pre-wrap;
                word-break: break-word;
            }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <h1>SecureRoute-AI</h1>
                <p>
                    Text-based OpenEnv environment for PII redaction and compliance routing.
                    Use the endpoints below to interact with the environment.
                </p>
                <ul>
                    <li><a href="/health"><code>GET /health</code></a> - service status</li>
                    <li><a href="/state"><code>GET /state</code></a> - current observation (after reset)</li>
                    <li><a href="/docs"><code>POST /reset</code></a> - initialize random or selected ticket (use docs or button below)</li>
                    <li><a href="/docs"><code>POST /step</code></a> - submit redaction and routing action (use docs)</li>
                    <li><a href="/docs"><code>GET /docs</code></a> - interactive API docs</li>
                </ul>
                <button class="btn" onclick="quickReset()">Quick Reset (POST /reset)</button>
                <pre id="reset-output">Click the button to test POST /reset.</pre>
                <p style="margin-top:14px;">
                    Space repo: <a href="https://huggingface.co/spaces/Chintamani007/secure-route-ai">huggingface.co/spaces/Chintamani007/secure-route-ai</a>
                </p>
            </div>
        </div>
        <script>
            async function quickReset() {
                const output = document.getElementById('reset-output');
                output.textContent = 'Loading...';
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const data = await response.json();
                    output.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    output.textContent = 'Request failed: ' + String(err);
                }
            }
        </script>
    </body>
</html>
"""
        )


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
