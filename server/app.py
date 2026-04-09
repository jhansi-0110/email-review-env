import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from models import EmailAction, EmailObservation
from server.environment import EmailReviewEnvironment

app = FastAPI(title="Email Review Environment")

_sessions = {}
_lock = threading.Lock()
DEFAULT = "default"


class StepRequest(BaseModel):
    action: EmailAction
    session_id: Optional[str] = DEFAULT


class ResetRequest(BaseModel):
    session_id: Optional[str] = DEFAULT


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    sid = req.session_id if req else DEFAULT
    with _lock:
        env = EmailReviewEnvironment()
        _sessions[sid] = env
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
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
    obs = env.step(req.action)
    done = bool(obs.done)
    if done:
        with _lock:
            _sessions.pop(sid, None)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward),
        "done": done,
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
    return {"episode_id": s.episode_id, "step_count": s.step_count}


@app.get("/tasks")
def list_tasks():
    from server.environment import TASKS
    return {"tasks": [{"id": t["id"], "difficulty": t["difficulty"]} for t in TASKS]}
