import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

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
    """
    Return all 3 tasks with grader info.
    Format matches OpenEnv validator expectations.
    """
    return {
        "tasks": [
            {
                "id":          t["id"],
                "name":        t["id"].replace("_", " ").title(),
                "difficulty":  t["difficulty"],
                "description": t["task_description"],
                "has_grader":  True,
                "grader_type": "deterministic",
                "score_range": {"min": 0.0, "max": 1.0},
                "action_space": {
                    "category":    ["billing", "technical", "general", "complaint"],
                    "priority":    ["low", "medium", "high", "urgent"],
                    "reply_draft": "str",
                },
                "observation_space": {
                    "email_subject": "str",
                    "email_body":    "str",
                    "sender_name":   "str",
                    "last_score":    "float",
                    "done":          "bool",
                    "reward":        "float",
                },
            }
            for t in TASKS
        ],
        "total": len(TASKS),
        "graders_available": len(TASKS),
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    sid     = req.session_id if req else DEFAULT
    task_id = req.task_id    if req else None

    with _lock:
        env = EmailReviewEnvironment()
        _sessions[sid] = env

    obs = env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward":      0.0,
        "done":        False,
        "task_id":     task_id,
    }


@app.post("/step")
def step(req: StepRequest):
    sid = req.session_id or DEFAULT
    with _lock:
        if sid not in _sessions:
            env = EmailReviewEnvironment()
            env.reset()
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
        "info":        {"score": float(obs.reward), "task_completed": done},
    }


@app.get("/state")
def state(session_id: str = DEFAULT):
    with _lock:
        if session_id not in _sessions:
            env = EmailReviewEnvironment()
            env.reset()
            _sessions[session_id] = env
        env = _sessions[session_id]
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
    }


@app.post("/grade")
def grade(task_id: str, req: StepRequest):
    """Direct grade endpoint — grade an action for a specific task without state."""
    task = None
    for t in TASKS:
        if t["id"] == task_id:
            task = t
            break
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    score, breakdown = grade_action(task, req.action)
    return {
        "task_id":   task_id,
        "score":     score,
        "reward":    score,
        "breakdown": breakdown,
        "done":      True,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
