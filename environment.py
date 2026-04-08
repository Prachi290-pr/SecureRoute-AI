import json
import random
from typing import Dict, Any, Tuple
from models import Observation, Action, Reward

class SecureRouteEnv:
    def __init__(self, dataset_path: str = "tickets.json"):
        self.dataset_path = dataset_path
        self.tickets = self._load_data()
        self.current_ticket = None

    def _load_data(self) -> list[dict]:
        try:
            with open(self.dataset_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset {self.dataset_path} not found. Ensure it is in the root directory.")

    def reset(self, ticket_id: int = None) -> Observation:
        """Resets the environment and loads a specific or random ticket."""
        if ticket_id is not None:
            self.current_ticket = next((t for t in self.tickets if t["id"] == ticket_id), None)
            if not self.current_ticket:
                raise ValueError(f"Ticket ID {ticket_id} not found.")
        else:
            self.current_ticket = random.choice(self.tickets)

        return self.state()

    def state(self) -> Observation:
        """Returns the current state of the environment."""
        if not self.current_ticket:
            raise ValueError("Environment must be reset before calling state().")

        return Observation(
            ticket_id=self.current_ticket["id"],
            text=self.current_ticket["text"]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Evaluates the agent's action and returns the reward."""
        if not self.current_ticket:
            raise ValueError("Environment must be reset before taking a step.")

        true_routing = self.current_ticket["true_routing"]
        contains_pii = self.current_ticket["contains_pii"]
        pii_to_redact = self.current_ticket["pii_to_redact"]
        original_text = self.current_ticket["text"]

        score = 0.0
        reasoning = []

        # 1. HARD FAIL: Check for PII Leakage
        if contains_pii and pii_to_redact in action.redacted_text:
            reward = Reward(score=0.0, reason="CRITICAL FAILURE: PII leaked in the action text.")
            return self.state(), reward, True, {"error": "pii_leak"}

        # 2. Evaluate Routing (+0.3)
        if action.routing.value == true_routing:
            score += 0.3
            reasoning.append("Correct routing (+0.3).")
        else:
            reasoning.append(f"Incorrect routing. Expected {true_routing}, got {action.routing.value}.")

        # 3. Evaluate Redaction (+0.7)
        if contains_pii:
            expected_redacted_text = original_text.replace(pii_to_redact, "[REDACTED]")
            if action.redacted_text == expected_redacted_text:
                score += 0.7
                reasoning.append("Perfect redaction (+0.7).")
            else:
                reasoning.append("Imperfect redaction: Text was altered incorrectly or PII was not replaced with exactly '[REDACTED]'.")
        else:
            # If no PII exists, the text should not be altered
            if action.redacted_text == original_text:
                score += 0.7
                reasoning.append("No PII present, text correctly left unaltered (+0.7).")
            else:
                reasoning.append("Text was altered unnecessarily.")

        reward = Reward(score=round(score, 1), reason=" ".join(reasoning))
        done = True # Triage is a single-turn action

        return self.state(), reward, done, {}