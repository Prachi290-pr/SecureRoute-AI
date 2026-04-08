from environment import SecureRouteEnv
from models import Action

EASY_TICKET_ID = 1
MEDIUM_TICKET_ID = 3
HARD_TICKET_ID = 10

def grade_task(ticket_id: int, agent_action: Action) -> float:
    """Instantiates the environment, runs the specific ticket, and returns the score."""
    env = SecureRouteEnv()
    env.reset(ticket_id=ticket_id)
    _, reward, _, _ = env.step(agent_action)

    return reward.score

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