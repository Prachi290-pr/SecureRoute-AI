from pydantic import BaseModel, Field
from enum import Enum

class RoutingDepartment(str, Enum):
    IT = "IT"
    BILLING = "BILLING"
    SECURITY = "SECURITY"

class Observation(BaseModel):
    ticket_id: int
    text: str = Field(description="The original customer support ticket text to be processed.")

class Action(BaseModel):
    redacted_text: str = Field(description="The ticket text with PII perfectly replaced by '[REDACTED]'.")
    routing: RoutingDepartment = Field(description="The department the ticket should be routed to.")

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="The total reward score.")
    reason: str = Field(description="Explanation of the assigned score for logging purposes.")