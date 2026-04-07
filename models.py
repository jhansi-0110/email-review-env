"""
models.py
---------
Defines the "shapes" of data flowing between the AI agent and the environment.

- EmailAction  : what the AI sends (its answer/response)
- EmailObservation : what the environment sends back (the email + feedback)

Think of these as typed forms — every field is required and validated.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class EmailAction(Action):
    """
    What the AI agent submits for each email task.

    The agent reads an email and must:
    1. Categorize it  (billing / technical / general / complaint)
    2. Prioritize it  (low / medium / high / urgent)
    3. Draft a reply
    """
    category: str = Field(
        ...,
        description="Email category. One of: billing, technical, general, complaint"
    )
    priority: str = Field(
        ...,
        description="Priority level. One of: low, medium, high, urgent"
    )
    reply_draft: str = Field(
        ...,
        description="The drafted reply to send back to the customer"
    )


class EmailObservation(Observation):
    """
    What the environment sends to the AI agent.

    Contains the email to process AND feedback on the last action taken.
    On reset(), feedback fields are empty — just the first email is shown.
    """
    # The email the agent needs to handle
    email_subject: str = Field(..., description="Subject line of the email")
    email_body: str = Field(..., description="Full body text of the email")
    sender_name: str = Field(..., description="Name of the customer who sent the email")

    # Feedback from the environment (filled after each step)
    task_description: str = Field(
        default="",
        description="Description of what the agent should do"
    )
    last_score: float = Field(
        default=0.0,
        description="Score for the last action (0.0 to 1.0)"
    )
    score_breakdown: str = Field(
        default="",
        description="Human-readable breakdown of how the score was calculated"
    )
    task_completed: bool = Field(
        default=False,
        description="Whether all tasks have been completed"
    )
