from environment import SecureRouteEnv
from models import Action
import math

EASY_TICKET_ID = 1
MEDIUM_TICKET_ID = 3
HARD_TICKET_ID = 10
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def make_meta_safe(score):
    try:
        s = float(score)
    except Exception:
        return 0.01
    
    if s >= 1.0:
        return 0.99
    
    if s <= 0.0:
        return 0.01
        
    return s

def grade_task(ticket_id: int, agent_action: Action) -> float:
    """Instantiates the environment, runs the specific ticket, and returns the score."""
    try:
        env = SecureRouteEnv()
        env.reset(ticket_id=ticket_id)
        _, reward, _, info = env.step(agent_action)
        normalized = make_meta_safe(reward.score)
        print(f"[GRADER] ticket={ticket_id} raw_score={reward.score} normalized_score={normalized} info={info}")
        return normalized
    except Exception as exc:
        # Fallback keeps score inside validator-required range and provides debug signal.
        print(f"[GRADER] ticket={ticket_id} error={exc}")
        return 0.01

# Easy Task: Normal IT ticket (No PII)
# Using Ticket ID 1: SharePoint Access
def grade_easy(*args, **kwargs):
    if args:
        agent_action = args[0]
    elif "agent_action" in kwargs:
        agent_action = kwargs["agent_action"]
    elif "action" in kwargs:
        agent_action = kwargs["action"]
    else:
        agent_action = None
    return grade_task(ticket_id=EASY_TICKET_ID, agent_action=agent_action)

# Medium Task: Billing ticket with 16-digit Credit Card
# Using Ticket ID 3: Double Charged Invoice
def grade_medium(*args, **kwargs):
    if args:
        agent_action = args[0]
    elif "agent_action" in kwargs:
        agent_action = kwargs["agent_action"]
    elif "action" in kwargs:
        agent_action = kwargs["action"]
    else:
        agent_action = None
    return grade_task(ticket_id=MEDIUM_TICKET_ID, agent_action=agent_action)

# Hard Task: Security ticket with phishing attempt and SSN
# Using Ticket ID 10: Phishing email received
def grade_hard(*args, **kwargs):
    if args:
        agent_action = args[0]
    elif "agent_action" in kwargs:
        agent_action = kwargs["agent_action"]
    elif "action" in kwargs:
        agent_action = kwargs["action"]
    else:
        agent_action = None
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