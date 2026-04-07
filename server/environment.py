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
        "task_description": "Task 1 (Easy): Billing double charge. correct_category=billing, correct_priority=high.",
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
        "task_description": "Task 2 (Medium): Angry premium subscriber. correct_category=complaint, correct_priority=urgent.",
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
            "Hello, I am facing two separate issues.\n\n"
            "1. TECHNICAL: API calls failing with error 401 after rotating keys. "
            "Invalid signature error. Tried regenerating twice.\n\n"
            "2. BILLING: Should be on Enterprise tier 15000 per month but invoice "
            "shows Business tier 7500 per month. SLA not active.\n\n"
            "50 engineers blocked. CTO aware. Account: ENT-00291."
        ),
        "task_description": "Task 3 (Hard): API 401 + wrong billing tier. correct_category=technical, correct_priority=urgent.",
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
    rl = action.reply_draft.lower()

    cat_correct = task['correct_category']
    pri_correct = task['correct_priority']
    keywords = task['required_keywords']
    forbidden = task['forbidden_phrases']
    min_len = task['min_reply_length']

    if action.category.lower().strip() == cat_correct:
        score += 0.25
        notes.append('OK Category correct ' + action.category + ': +0.25')
    else:
        notes.append('WRONG Category got ' + action.category + ' expected ' + cat_correct + ': +0.00')

    if action.priority.lower().strip() == pri_correct:
        score += 0.25
        notes.append('OK Priority correct ' + action.priority + ': +0.25')
    else:
        notes.append('WRONG Priority got ' + action.priority + ' expected ' + pri_correct + ': +0.00')

    found = [kw for kw in keywords if kw.lower() in rl]
    kws = round(len(found) / max(len(keywords), 1) * 0.30, 3)
    score += kws
    notes.append('Keywords ' + str(len(found)) + '/' + str(len(keywords)) + ' ' + str(found) + ': +' + str(kws))

    bad = [fp for fp in forbidden if fp.lower() in rl]
    if not bad:
        score += 0.10
        notes.append('OK No forbidden phrases: +0.10')
    else:
        notes.append('WRONG Forbidden found ' + str(bad) + ': +0.00')

    wc = len(action.reply_draft.split())
    if wc >= min_len:
        score += 0.10
        notes.append('OK Length OK ' + str(wc) + ' words: +0.10')
    else:
        notes.append('WRONG Too short ' + str(wc) + '/' + str(min_len) + ' words: +0.00')

    score = round(min(score, 1.0), 3)
    breakdown = chr(10).join(notes) + chr(10) + 'FINAL SCORE: ' + str(score)
    return score, breakdown


class EmailReviewEnvironment(Environment):
    """Email Triage Environment with 3 tasks Easy/Medium/Hard."""

    def __init__(self):
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._task_index = 0
        self._scores = []
        self._state = State(episode_id=self._episode_id, step_count=0)

    def reset(self) -> EmailObservation:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._task_index = 0
        self._scores = []
        self._state = State(episode_id=self._episode_id, step_count=0)
        t = TASKS[0]
        return EmailObservation(
            email_subject=t['email_subject'],
            email_body=t['email_body'],
            sender_name=t['sender_name'],
            task_description=t['task_description'],
            last_score=0.0,
            score_breakdown='',
            task_completed=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: EmailAction) -> EmailObservation:
        self._step_count += 1
        self._state = State(episode_id=self._episode_id, step_count=self._step_count)

        if self._task_index >= len(TASKS):
            return EmailObservation(
                email_subject='Done',
                email_body='All tasks complete. Call reset().',
                sender_name='System',
                task_description='',
                last_score=0.0,
                score_breakdown='Already finished.',
                task_completed=True,
                done=True,
                reward=0.0,
            )

        t = TASKS[self._task_index]
        score, breakdown = grade_action(t, action)
        self._scores.append(score)
        self._task_index += 1
        finished = (self._task_index >= len(TASKS))

        if finished:
            avg = round(sum(self._scores) / len(self._scores), 3)
            return EmailObservation(
                email_subject='All tasks complete',
                email_body='Scores: ' + str(self._scores) + ' Average: ' + str(avg),
                sender_name='System',
                task_description='Episode complete. Call reset() to restart.',
                last_score=score,
                score_breakdown=breakdown,
                task_completed=True,
                done=True,
                reward=score,
            )

        nt = TASKS[self._task_index]
        return EmailObservation(
            email_subject=nt['email_subject'],
            email_body=nt['email_body'],
            sender_name=nt['sender_name'],
            task_description=nt['task_description'],
            last_score=score,
            score_breakdown=breakdown,
            task_completed=False,
            done=False,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state