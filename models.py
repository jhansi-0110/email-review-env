from typing import Optional
from pydantic import BaseModel, Field


class EmailAction(BaseModel):
    category: str = Field(..., description="billing | technical | general | complaint")
    priority: str = Field(..., description="low | medium | high | urgent")
    reply_draft: str = Field(..., description="Professional reply to the customer")


class EmailObservation(BaseModel):
    email_subject: str = Field(..., description="Subject line of the email")
    email_body: str = Field(..., description="Full body of the email")
    sender_name: str = Field(..., description="Name of the customer")
    task_description: str = Field(default="", description="What the agent should do")
    last_score: float = Field(default=0.0, description="Score for previous action")
    score_breakdown: str = Field(default="", description="Breakdown of scoring")
    task_completed: bool = Field(default=False)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
