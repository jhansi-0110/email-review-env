
"""
server/app.py
Custom FastAPI server that manages environment sessions properly.
Each reset() creates a brand new session with a unique session_id.
Each step() operates on the correct session.
"""
import sys, os, uuid, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import EmailAction, EmailObservation
from server.environment import EmailReviewEnvironment

app = FastAPI(title="Email Review Environment")

# ── Global session store ──────────────────────────────────────────
# Maps session_id -> EmailReviewEnvironment instance
_sessions = {}
_lock = threading.Lock()
_default_session = "default"

def get_or_create_session(session_id: str = _default_session) -> EmailReviewEnvironment:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = EmailReviewEnvironment()
        return _sessions[session_id]


# ── Request/Response models ───────────────────────────────────────

class StepRequest(BaseModel):
    action: EmailAction
    session_id: Optional[str] = _default_session

class ResetRequest(BaseModel):
    session_id: Optional[str] = _default_session


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    session_id = req.session_id if req else _default_session
    with _lock:
        # Always create FRESH instance on reset
        env = EmailReviewEnvironment()
        _sessions[session_id] = env
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/step")
def step(req: StepRequest):
    session_id = req.session_id or _default_session
    with _lock:
        if session_id not in _sessions:
            # Auto-create if not exists
            env = EmailReviewEnvironment()
            env.reset()
            _sessions[session_id] = env
        env = _sessions[session_id]

    obs = env.step(req.action)
    done = obs.done

    # Clean up finished sessions
    if done:
        with _lock:
            _sessions.pop(session_id, None)

    return {
        "observation": obs.model_dump(),
        "reward":      obs.reward,
        "done":        done,
    }


@app.get("/state")
def state(session_id: str = _default_session):
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


@app.get("/tasks")
def list_tasks():
    """List all tasks — required by OpenEnv spec."""
    from server.environment import TASKS
    return {
        "tasks": [
            {
                "id":         t["id"],
                "difficulty": t["difficulty"],
                "description":t["task_description"],
            }
            for t in TASKS
        ]
    }
