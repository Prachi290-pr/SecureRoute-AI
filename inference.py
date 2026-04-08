import os
from openai import OpenAI
from environment import SecureRouteEnv
from models import Action


def load_api_key() -> str:
    # Priority 1: environment variable
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    # Priority 2: file path (customizable via OPENAI_API_KEY_FILE)
    key_file = os.getenv("OPENAI_API_KEY_FILE", ".secrets/openai_api_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            file_key = f.read().strip()
            if file_key and "PASTE_YOUR_OPENAI_API_KEY_HERE" not in file_key:
                return file_key

    raise ValueError(
        "Missing OpenAI API key. Set OPENAI_API_KEY or add your key to .secrets/openai_api_key.txt"
    )

# 1. Load Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") # Swap to whatever model you are using
HF_TOKEN = os.getenv("HF_TOKEN", "") # Keep for HF Space deployment

client = OpenAI(api_key=load_api_key(), base_url=API_BASE_URL)

def run_inference(ticket_id: int, task_name: str):
    print(f"[START] Episode {ticket_id} - Task: {task_name}")

    env = SecureRouteEnv()
    obs = env.reset(ticket_id=ticket_id)

    system_prompt = """
    You are SecureRouteAI, a strict compliance triage agent.
    Analyze the customer support ticket.
    1. Replace ANY 16-digit credit card number or 9-digit SSN (XXX-XX-XXXX) with exactly "[REDACTED]". Do not alter any other text.
    2. Route the ticket to IT, BILLING, or SECURITY.
    Output strictly in JSON matching the requested schema.
    """

    try:
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Ticket:\n{obs.text}"}
            ],
            response_format=Action,
        )

        agent_action = response.choices[0].message.parsed

        # Take the step in the environment
        next_obs, reward, done, info = env.step(agent_action)

        print(f"[STEP] Obs: {obs.ticket_id} | Action: {agent_action.routing.value} / PII Handled | Reward: {reward.score}")
        print(f"[END] Final Score: {reward.score}\n")

    except Exception as e:
        print(f"[STEP] Error during inference: {str(e)}")
        print(f"[END] Final Score: 0.0\n")

if __name__ == "__main__":
    # Test all three deterministic tasks
    run_inference(ticket_id=1, task_name="EASY")
    run_inference(ticket_id=3, task_name="MEDIUM")
    run_inference(ticket_id=8, task_name="HARD")