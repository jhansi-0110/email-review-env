"""
server/app.py
-------------
Creates the FastAPI web server that hosts the environment.
OpenEnv's create_app() handles all the WebSocket endpoints automatically.
You just need to pass it your Environment class, Action, and Observation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app
from models import EmailAction, EmailObservation
from server.environment import EmailReviewEnvironment

# Pass the CLASS (not an instance) — OpenEnv creates a fresh instance
# for each connected session automatically.
app = create_app(
    EmailReviewEnvironment,
    EmailAction,
    EmailObservation,
    env_name="email_review_env"
)
