"""
server/environment.py — Email Triage OpenEnv Environment
3 tasks: Easy → Medium → Hard
Scores: 0.0 to 1.0 with partial credit
"""

import re
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EmailAction, EmailObservation

TASKS = [
    {
        "id": "task_1_easy",
        "difficulty": "easy",
        "sender_name": "Priya Sharma",
        "email_subject": "Invoice question",
        "email_body": (
            "Hi, I received my invoice for this month and noticed I was charged "
            "twice for the same subscription. The amount is 499 x 2 = 998. "
            "Could you please look into this and refund the extra charge? "
            "My account ID is #84721. Thank you."
        ),
        "task_description": (
            "Task 1 (Easy): Categorize this email correctly, set the right priority, "
            "and draft a polite reply that acknowledges the double charge issue, "
            "mentions the account ID, and promises a resolution."
        ),
        "correct_category": "billing",
        "correct_priority": "high",
        "required_keywords": ["refund", "account", "apologize"],
        "forbidden_phrases": ["cannot help", "not our problem"],
        "min_reply_length": 40,
    },
    {
        "id": "task_2_medium",
        "difficulty": "medium",
        "sender_name": "Ravi Menon",
        "email_subject": "EXTREMELY FRUSTRATED - service has been down for 3 days!!!",
        "email_body": (
            "I am absolutely furious. Your service has been completely down for "
            "THREE DAYS and no one has responded to any of my support tickets. "
            "I am a premium subscriber paying 2999 per month. This is completely "
            "unacceptable. I want a full refund for this month AND compensation. "
            "If this is not resolved TODAY I am cancelling my account and posting "
            "reviews everywhere. This is a DISASTER for my business."
        ),
        "task_description": (
            "Task 2 (Medium): Angry premium customer with service outage. "
            "Category is complaint, priority is urgent. Draft a reply that "
            "de-escalates, acknowledges frustration, and offers concrete next steps."
        ),
        "correct_category": "complaint",
        "correct_priority": "urgent",
        "required_keywords": ["apologize", "escalat", "premium", "resolve"],
        "forbidden_phrases": ["we understand your frustration", "valued customer", "at your earliest convenience"],
        "min_reply_length": 60,
    },
    {
        "id": "task_3_hard",
        "difficulty": "hard",
        "sender_name": "Ananya Krishnan",
        "email_subject": "API authentication failing + wrong billing tier",
        "email_body": (
            "Hello, I am facing two separate issues:\n\n"
            "1. TECHNICAL: Our API calls have been failing with error 401 since "
            "yesterday after we rotated our API keys. The new key returns Invalid signature "
            "even though we followed the docs exactly. We have tried regenerating twice.\n\n"
            "2. BILLING: We were supposed to be on the Enterprise tier (15000 per month) "
            "but our invoice shows Business tier (7500 per month). "
            "Our SLA guarantees and dedicated support are also not active.\n\n"
            "We need both fixed urgently. Our team of 50 engineers are blocked. "
            "CTO is aware. Account: ENT-00291."
        ),
        "task_description": (
            "Task 3 (Hard): Complex dual-issue email with both technical (API 401) "
            "and billing (wrong tier) problems. Category technical, priority urgent. "
            "Reply MUST address BOTH issues, include account number, mention specific "
            "technical steps AND billing escalation."
        ),
        "correct_category": "technical",
        "correct_priority": "urgent",
        "required_keywords": ["401", "api", "billing", "enterprise", "account", "escalat"],
        "forbidden_phrases": ["we cannot", "not possible"],
        "min_reply_length": 100,
    },
]


def grade_action(task: dict, action: EmailAction) -> tuple:
    score = 0.0
    notes = []

    # 1. Category (0.25)
    if action.category.lower().strip() == task["correct_category"]:
        score += 0.25
        notes.append(f"✅ Category correct ({action.category}): +0.25")
    else:
        notes.append(f"❌ Category wrong — got {action.category!r}, expected {task['correct_category']!r}: +0.00")

    # 2. Priority (0.25)
    if action.priority.lower().strip() == task["correct_priority"]:
        score += 0.25
        notes.append(f"✅ Priority correct ({action.priority}): +0.25")
    else:
        notes.append(f"❌ Priority wrong — got {action.priority!r}, expected {task['correct_priority']!r}: +0.00")

    # 3. Required keywords (0.30)
    reply_lower = action.reply_draft.lower()
    found = [kw for kw in task["required_keywords"] if kw.lower() in reply_lower]
    kw_score = round(len(found) / max(len(task["required_keywords"]), 1) * 0.30, 3)
    score += kw_score
    notes.append(f"{'✅' if len(found)==len(task['required_keywords']) else '⚠️'} Keywords found {len(found)}/{len(task['required_keywords'])} ({found}): +{kw_score}")

    # 4. No forbidden phrases (0.10)
    forbidden_found = [fp for fp in task["forbidden_phrases"] if fp.lower() in reply_lower]
    if not forbidden_found:
        score += 0.10
        notes.append("✅ No forbidden phrases: +0.10")
    else:
        notes.append(f"❌ Forbidden phrases found {forbidden_found}: +0.00")

    # 5. Reply length (0.10)
    word_count = len(action.reply_draft.split())
    if word_count >= task["min_reply_length"]:
        score += 0.10
        notes.append(f"✅ Reply length OK ({word_count} words): +0.10")
    else:
        notes.append(f"❌ Reply too short — {word_count} words, need {task['min_reply_length']}: +0.00")

    score = round(min(score, 1.0), 3)
    breakdown = "\n".join(notes) + f"\n\nFINAL SCORE: {score}"
    return score, breakdown


class EmailReviewEnvironment(Environment):
    """
    Email Triage and Response Environment.
    3 tasks: Easy, Medium, Hard.
    Scores 0.0 to 1.0 with partial credit.
    """

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._task_scores = []

    def reset(self) -> EmailObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._task_scores = []
        task = TASKS[0]
        return EmailObservation(
            email_subject=task["email_subject"],
            email_body=task["email_body"],
            sender_name=task["sender_name"],
            task_description=task["task_description"],
            last_score=0.0,
            score_breakdown="",
            task_completed=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailAction) -> EmailObservation:
        self._state.step_count += 1
        task = TASKS[self._current_task_index]

        score, breakdown = grade_action(task, action)
        self._task_scores.append(score)

        self._current_task_index += 1
        all_done = (self._current_task_index >= len(TASKS))

        if all_done:
            avg = round(sum(self._task_scores) / len(self._task_scores), 3)
            return EmailObservation(
                email_subject="All tasks complete",
                email_body=(
                    f"Episode finished! Task scores: {self._task_scores} "
                    f"Average: {avg}"
                ),
                sender_name="System",
                task_description="Episode complete. Call reset() to start again.",
                last_score=score,
                score_breakdown=breakdown,
                task_completed=True,
                done=True,        # ← EXPLICITLY True
                reward=score,
            )
        else:
            next_task = TASKS[self._current_task_index]
            return EmailObservation(
                email_subject=next_task["email_subject"],
                email_body=next_task["email_body"],
                sender_name=next_task["sender_name"],
                task_description=next_task["task_description"],
                last_score=score,
                score_breakdown=breakdown,
                task_completed=False,
                done=False,       # ← EXPLICITLY False
                reward=score,
            )

    @property
    def state(self) -> State:
        return self._state
