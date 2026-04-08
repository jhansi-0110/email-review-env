---
title: Email Review Envw Env
emoji: 💻
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---
# Email Review Environment 📧

A real-world **customer support email triage** environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

An AI agent learns to read customer emails and make three decisions:
1. **Categorize** the email (billing / technical / general / complaint)
2. **Prioritize** it (low / medium / high / urgent)
3. **Draft a professional reply**

It receives a reward score (0.0 to 1.0) with partial credit after each email.

---

## Environment Description

Every company receives thousands of support emails daily. Routing them to the right team and responding quickly is critical. This environment trains an AI agent to handle this task with increasing complexity across 3 tasks.

| Task | Difficulty | Scenario |
|------|-----------|----------|
| Task 1 | Easy | Simple billing double-charge complaint |
| Task 2 | Medium | Angry premium subscriber with 3-day outage |
| Task 3 | Hard | Dual-issue: API auth failure + wrong billing tier |

---

## Action Space

| Field | Type | Values |
|-------|------|--------|
| `category` | string | `billing`, `technical`, `general`, `complaint` |
| `priority` | string | `low`, `medium`, `high`, `urgent` |
| `reply_draft` | string | Professional reply (40-100+ words depending on task) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `email_subject` | string | Subject line of the email |
| `email_body` | string | Full email body text |
| `sender_name` | string | Customer name |
| `task_description` | string | Hints about what is expected |
| `last_score` | float | Score from the previous step (0.0-1.0) |
| `score_breakdown` | string | Detailed score explanation |
| `task_completed` | bool | True when all 3 tasks done |
| `done` | bool | True when episode is over |
| `reward` | float | Reward for this step |

## Reward Function

Each task is scored out of 1.0 with partial credit:

| Component | Points | How |
|-----------|--------|-----|
| Correct category | 0.25 | Exact match |
| Correct priority | 0.25 | Exact match |
| Required keywords in reply | 0.30 | Proportional (found/total) |
| No forbidden phrases | 0.10 | None of the bad phrases present |
| Reply meets minimum length | 0.10 | Word count above threshold |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop
- Hugging Face account + CLI

### 1. Install dependencies

```bash
pip install -r server/requirements.txt
```

### 2. Run server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://localhost:8000/health — should return {"status": "ok"}

### 3. Run tests locally

```bash
python test_local.py
```

### 4. Run inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:8000"

python inference.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | Hugging Face API token |
| `ENV_URL` | No | Environment URL (default: localhost:8000) |

---

## Project Structure

```
email_review_env/
├── models.py              # Action and Observation types
├── client.py              # Agent-side client
├── inference.py           # Main evaluation script
├── test_local.py          # Local testing script
├── openenv.yaml           # Environment metadata
├── pyproject.toml         # Project config
├── README.md              # This file
└── server/
    ├── environment.py     # Core logic: tasks, graders, rewards
    ├── app.py             # FastAPI server
    ├── requirements.txt   # Dependencies
    └── Dockerfile         # Container definition
```