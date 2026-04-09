import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from models import EmailAction, EmailObservation
from server.environment import EmailReviewEnvironment, TASKS, grade_action

app = FastAPI(title="Email Review Environment")

_sessions = {}
_lock = threading.Lock()
DEFAULT = "default"


class StepRequest(BaseModel):
    action: EmailAction
    session_id: Optional[str] = DEFAULT


class ResetRequest(BaseModel):
    session_id: Optional[str] = DEFAULT
    task_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return all tasks with grader info — required by OpenEnv spec."""
    return {
        "tasks": [
            {
                "id":          t["id"],
                "difficulty":  t["difficulty"],
                "description": t["task_description"],
                "grader":      "deterministic",
                "score_range": [0.0, 1.0],
                "action_space": {
                    "category":    ["billing", "technical", "general", "complaint"],
                    "priority":    ["low", "medium", "high", "urgent"],
                    "reply_draft": "string (min 40 words)",
                },
            }
            for t in TASKS
        ]
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    sid     = req.session_id if req else DEFAULT
    task_id = req.task_id    if req else None

    with _lock:
        env = EmailReviewEnvironment()
        # If a specific task_id is requested, start from that task
        if task_id:
            for i, t in enumerate(TASKS):
                if t["id"] == task_id:
                    env._task_index = i
                    break
        _sessions[sid] = env

    obs = env.reset_to_current()
    return {
        "observation": obs.model_dump(),
        "reward":      0.0,
        "done":        False,
    }


@app.post("/step")
def step(req: StepRequest):
    sid = req.session_id or DEFAULT
    with _lock:
        if sid not in _sessions:
            env = EmailReviewEnvironment()
            env.reset_to_current()
            _sessions[sid] = env
        env = _sessions[sid]

    obs  = env.step(req.action)
    done = bool(obs.done)

    if done:
        with _lock:
            _sessions.pop(sid, None)

    return {
        "observation": obs.model_dump(),
        "reward":      float(obs.reward),
        "done":        done,
    }


@app.get("/state")
def state(session_id: str = DEFAULT):
    with _lock:
        if session_id not in _sessions:
            env = EmailReviewEnvironment()
            env.reset_to_current()
            _sessions[session_id] = env
        env = _sessions[session_id]
    s = env.state
    return {"episode_id": s.episode_id, "step_count": s.step_count}


@app.post("/grade")
def grade(req: StepRequest):
    """Direct grader endpoint — grade an action for a specific task."""
    sid = req.session_id or DEFAULT
    with _lock:
        if sid not in _sessions:
            env = EmailReviewEnvironment()
            env.reset_to_current()
            _sessions[sid] = env
        env = _sessions[sid]

    idx = min(env._task_index, len(TASKS) - 1)
    task = TASKS[idx]
    score, breakdown = grade_action(task, req.action)
    return {
        "task_id":   task["id"],
        "score":     score,
        "breakdown": breakdown,
        "done":      True,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
