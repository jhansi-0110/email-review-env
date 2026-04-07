
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
            "Task 1 (Easy): Categorize this email, set priority, and draft a reply "
            "acknowledging the double charge and promising a refund."
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
            "Category complaint, priority urgent. De-escalate and offer next steps."
        ),
        "correct_category": "complaint",
        "correct_priority": "urgent",
        "required_keywords": ["apologize", "escalat", "premium", "resolve"],
        "forbidden_phrases": ["we understand your frustration", "valued customer"],
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
            "yesterday after we rotated our API keys. Invalid signature error.\n\n"
            "2. BILLING: We were supposed to be on Enterprise tier (15000/month) "
            "but invoice shows Business tier (7500/month). SLA not active.\n\n"
            "50 engineers blocked. CTO aware. Account: ENT-00291."
        ),
        "task_description": (
            "Task 3 (Hard): Dual-issue email — API 401 + wrong billing tier. "
            "Category technical, priority urgent. Address BOTH issues in reply."
        ),
        "correct_category": "technical",
        "correct_priority": "urgent",
        "required_keywords": ["401", "api", "billing", "enterprise", "account", "escalat"],
        "forbidden_phrases": ["we cannot", "not possible"],
        "min_reply_length": 80,
    },
]


def grade_action(task, action):
    score = 0.0
    notes = []
    reply_lower = action.reply_draft.lower()

    if action.category.lower().strip() == task["correct_category"]:
        score += 0.25
        notes.append(f"✅ Category correct ({action.category}): +0.25")
    else:
        notes.append(f"❌ Category wrong — got {action.category!r}, expected {task['correct_category']!r}: +0.00")

    if action.priority.lower().strip() == task["correct_priority"]:
        score += 0.25
        notes.append(f"✅ Priority correct ({action.priority}): +0.25")
    else:
        notes.append(f"❌ Priority wrong — got {action.priority!r}, expected {task['correct_priority']!r}: +0.00")

    found = [kw for kw in task["required_keywords"] if kw.lower() in reply_lower]
    kw_score = round(len(found) / max(len(task["required_keywords"]), 1) * 0.30, 3)
    score += kw_score
    notes.append(f"{'✅' if len(found)==len(task['required_keywords']) else '⚠️'} Keywords {len(found)}/{len(task['required_keywords'])} ({found}): +{kw_score}")

    forbidden_found = [fp for fp in task["forbidden_phrases"] if fp.lower() in reply_lower]
    if not forbidden_found:
        score += 0.10
        notes.append("✅ No forbidden phrases: +0.10")
    else:
        notes.append(f"❌ Forbidden found {forbidden_found}: +0.00")

    wc = len(action.reply_draft.split())
    if wc >= task["min_reply_length"]:
        score += 0.10
        notes.append(f"✅ Reply length OK ({wc} words): +0.10")
    else:
        notes.append(f"❌ Reply too short — {wc} words, need {task['min_reply_length']}: +0.00")

    score = round(min(score, 1.0), 3)
    return score, "\n".join(notes) + f"\n\nFINAL SCORE: {score}"


class EmailReviewEnvironment(Environment):
    """
    Email Triage Environment — 3 tasks Easy/Medium/Hard.
    Each instance is independent (OpenEnv creates one per WebSocket session).
    """

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._current_task_index = 0
        self._task_scores = []
        self._state = State(episode_id=self._episode_id, step_count=0)

    def reset(self) -> EmailObservation:
        self._reset_state()
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
        self._step_count += 1
        self._state = State(episode_id=self._episode_id, step_count=self._step_count)

        # Guard: if already past all tasks, mark done
        if self._current_task_index >= len(TASKS):
            return EmailObservation(
                email_subject="Episode already complete",
                email_body="All tasks done. Please call reset() to start a new episode.",
                sender_name="System",
                task_description="Call reset() to restart.",
                last_score=0.0,
                score_breakdown="Episode already finished.",
                task_completed=True,
                done=True,
                reward=0.0,
            )

        task = TASKS[self._current_task_index]
        score, breakdown = grade_action(task, action)
        self._task_scores.append(score)
        self._current_task_index += 1

        is_last = (self._current_task_index >= len(TASKS))

        if is_last:
            avg = round(sum(self._task_scores) / len(self._task_scores), 3)
            return EmailObservation(
                email_subject="All tasks complete",
                email_body=f"Episode done. Scores: {self._task_scores}. Average: {avg}",
                sender_name="System",
                task_description="Episode complete. Call reset() to start again.",
                last_score=score,
                score_breakdown=breakdown,
                task_completed=True,
                done=True,
                reward=score,
            )

        next_task = TASKS[self._current_task_index]
        return EmailObservation(
            email_subject=next_task["email_subject"],
            email_body=next_task["email_body"],
            sender_name=next_task["sender_name"],
            task_description=next_task["task_description"],
            last_score=score,
            score_breakdown=breakdown,
            task_completed=False,
            done=False,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state
