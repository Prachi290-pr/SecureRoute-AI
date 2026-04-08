from environment import SecureRouteEnv
from models import Action, RoutingDepartment

def grade_task(ticket_id: int, agent_action: Action) -> float:
    """Instantiates the environment, runs the specific ticket, and returns the score."""
    env = SecureRouteEnv()
    env.reset(ticket_id=ticket_id)
    _, reward, _, _ = env.step(agent_action)

    print(f"Score: {reward.score} | Reason: {reward.reason}")
    return reward.score

# Easy Task: Normal IT ticket (No PII)
# Using Ticket ID 1: SharePoint Access
def grade_easy(agent_action: Action):
    print("[GRADER] Running Easy Task...")
    return grade_task(ticket_id=1, agent_action=agent_action)

# Medium Task: Billing ticket with 16-digit Credit Card
# Using Ticket ID 3: Double Charged Invoice
def grade_medium(agent_action: Action):
    print("[GRADER] Running Medium Task...")
    return grade_task(ticket_id=3, agent_action=agent_action)

# Hard Task: Security ticket with SSN
# Using Ticket ID 8: URGENT account compromise
def grade_hard(agent_action: Action):
    print("[GRADER] Running Hard Task...")
    return grade_task(ticket_id=8, agent_action=agent_action)