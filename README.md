---
title: Email Review Env
emoji: "📧"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Review Environment

An OpenEnv environment for customer support email triage.
The AI agent categorizes, prioritizes, and drafts replies to emails.

## Tasks
- Task 1 (Easy): Billing double charge
- Task 2 (Medium): Angry premium subscriber
- Task 3 (Hard): API failure + wrong billing tier

## API
- POST /reset
- POST /step
- GET /state
- GET /health

## Environment Variables Required
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
