from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

class RoutingDepartment(str, Enum):
    IT = "IT"
    BILLING = "BILLING"
    SECURITY = "SECURITY"

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: int | None = Field(
        default=None,
        description="Optional ticket identifier used by the benchmark and diagnostics.",
    )
    text: str = Field(description="The customer support ticket text to be triaged.")

class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redacted_text: str = Field(description="The ticket text with required PII replaced by '[REDACTED]'.")
    routing: RoutingDepartment = Field(description="The department the ticket should be routed to.")

class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0, description="Normalized episode reward in the range 0.0 to 1.0.")