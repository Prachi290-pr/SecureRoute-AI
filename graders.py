from environment import SecureRouteEnv
from models import Action

EASY_TICKET_ID = 1
MEDIUM_TICKET_ID = 3
HARD_TICKET_ID = 10
MIN_SCORE = 0.1
MAX_SCORE = 0.9


def to_validator_safe_interval(score: float) -> float:
    """Normalize score to a non-boundary range that remains valid after rounding."""
    raw = float(score)
    if raw <= 0.0:
        return MIN_SCORE
    if raw >= 1.0:
        return MAX_SCORE
    return min(MAX_SCORE, max(MIN_SCORE, round(raw, 2)))

def grade_task(ticket_id: int, agent_action: Action) -> float:
    """Instantiates the environment, runs the specific ticket, and returns the score."""
    try:
        env = SecureRouteEnv()
        env.reset(ticket_id=ticket_id)
        _, reward, _, info = env.step(agent_action)
        normalized = to_validator_safe_interval(reward.score)
        print(f"[GRADER] ticket={ticket_id} raw_score={reward.score} normalized_score={normalized} info={info}")
        return normalized
    except Exception as exc:
        # Fallback keeps score inside validator-required range and provides debug signal.
        print(f"[GRADER] ticket={ticket_id} error={exc}")
        return MIN_SCORE

# Easy Task: Normal IT ticket (No PII)
# Using Ticket ID 1: SharePoint Access
def grade_easy(agent_action: Action):
    return grade_task(ticket_id=EASY_TICKET_ID, agent_action=agent_action)

# Medium Task: Billing ticket with 16-digit Credit Card
# Using Ticket ID 3: Double Charged Invoice
def grade_medium(agent_action: Action):
    return grade_task(ticket_id=MEDIUM_TICKET_ID, agent_action=agent_action)

# Hard Task: Security ticket with phishing attempt and SSN
# Using Ticket ID 10: Phishing email received
def grade_hard(agent_action: Action):
    return grade_task(ticket_id=HARD_TICKET_ID, agent_action=agent_action)


if __name__ == "__main__":
    from models import RoutingDepartment

    env = SecureRouteEnv()

    obs_easy = env.reset(ticket_id=EASY_TICKET_ID)
    action_easy = Action(redacted_text=obs_easy.text, routing=RoutingDepartment.IT)

    obs_medium = env.reset(ticket_id=MEDIUM_TICKET_ID)
    action_medium = Action(
        redacted_text=obs_medium.text.replace("4111-1111-1111-1111", "[REDACTED]"),
        routing=RoutingDepartment.BILLING,
    )

    obs_hard = env.reset(ticket_id=HARD_TICKET_ID)
    action_hard = Action(
        redacted_text=obs_hard.text.replace("987-65-4321", "[REDACTED]"),
        routing=RoutingDepartment.SECURITY,
    )

    print("easy=", grade_easy(action_easy))
    print("medium=", grade_medium(action_medium))
    print("hard=", grade_hard(action_hard))