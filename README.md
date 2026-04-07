---
title: Email Review Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Review Environment

A real-world OpenEnv environment for customer support email triage.
An AI agent learns to categorize, prioritize, and draft replies to customer emails.

## Environment Description

The agent processes customer support emails one at a time.
For each email it must correctly categorize it, set the right priority,
and draft a professional empathetic reply.
The environment grades each response with partial credit scoring.

## Action Space
```python
EmailAction(
    category    = "billing",      # billing | technical | general | complaint
    priority    = "high",         # low | medium | high | urgent
    reply_draft = "Dear customer..." # minimum 40 words
)
```

## Observation Space
```python
EmailObservation(
    email_subject    = "Invoice question",
    email_body       = "I was charged twice...",
    sender_name      = "Priya Sharma",
    task_description = "Task 1 (Easy): ...",
    last_score       = 0.75,
    score_breakdown  = "Category correct: +0.25...",
    task_completed   = False,
    done             = False,
    reward           = 0.75
)
```

## Tasks

| Task | Difficulty | Scenario | Category | Priority |
|------|-----------|----------|----------|----------|
| 1 | Easy | Double billing charge | billing | high |
| 2 | Medium | Angry premium subscriber, 3-day outage | complaint | urgent |
| 3 | Hard | API auth failure + wrong billing tier | technical | urgent |

## Reward / Scoring (0.0 to 1.0 with partial credit)

| Component | Points |
|-----------|--------|
| Category correct | 0.25 |
| Priority correct | 0.25 |
| Required keywords in reply | 0.30 |
| No forbidden cliche phrases | 0.10 |
| Reply meets minimum length | 0.10 |

## Setup
```bash
pip install openenv-core fastapi uvicorn pydantic openai websockets
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Inference
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_URL="http://localhost:7860"
python inference.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| API_BASE_URL | Yes | https://api-inference.huggingface.co/v1 | LLM API endpoint |
| MODEL_NAME | Yes | meta-llama/Llama-3.1-8B-Instruct | Model identifier |
| HF_TOKEN | Yes | None | Hugging Face token |
| ENV_URL | No | http://localhost:7860 | Environment URL |

## Baseline Scores

Running `inference.py` with `meta-llama/Llama-3.1-8B-Instruct`:
- Task 1 (Easy): ~0.85
- Task 2 (Medium): ~0.65
- Task 3 (Hard): ~0.55
- Average: ~0.68

## Project Structure
email_review_env/
├── models.py              # Action and Observation type definitions
├── client.py              # EnvClient implementation
├── inference.py           # Baseline inference script
├── openenv.yaml           # Environment metadata
├── pyproject.toml         # Dependencies
├── README.md              # This file
└── server/
├── environment.py     # Core logic: tasks, graders, reward
├── app.py             # FastAPI server with session management
├── requirements.txt   # Server dependencies
└── Dockerfile         # Container definition
