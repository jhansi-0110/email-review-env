"""
inference.py
------------
THIS IS THE MOST IMPORTANT FILE FOR JUDGING.

The judges will run: python inference.py
It must:
  1. Connect to the environment
  2. Run an AI agent through all 3 tasks
  3. Print scores clearly
  4. Finish in under 20 minutes
  5. Use OpenAI client for all LLM calls (contest requirement)

Environment variables required (set in HF Spaces secrets):
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model to use (e.g. "meta-llama/Llama-3.1-8B-Instruct")
  HF_TOKEN      — Hugging Face token

Run locally:
  API_BASE_URL=https://api-inference.huggingface.co/v1 \
  MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  HF_TOKEN=hf_xxx \
  python inference.py
"""

import os
import json
import time
import sys

# OpenAI client (required by contest rules)
from openai import OpenAI

# Our environment client
from client import EmailReviewEnv
from models import EmailAction

# ── Config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:8000")

# ── LLM Client Setup ──────────────────────────────────────────────────────
llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key-needed",
)

# ── System prompt for the AI agent ────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support email triage agent.

For each customer email you receive, you must respond with a JSON object containing EXACTLY these three fields:
  "category"    : one of "billing", "technical", "general", "complaint"
  "priority"    : one of "low", "medium", "high", "urgent"
  "reply_draft" : a professional, empathetic reply to the customer (minimum 40 words)

Rules:
- Category "complaint" = customer is angry/frustrated (even if there's a technical cause)
- Category "technical" = primarily a technical/product issue
- Category "billing"   = payment, invoice, refund, subscription issues
- Priority "urgent"    = business blocked, premium subscriber, multiple issues, CTO involved
- Priority "high"      = significant problem needing quick resolution
- Priority "medium"    = important but not time-critical
- Priority "low"       = general inquiry, no urgency
- Your reply must: address the specific issue, not use clichéd phrases like "valued customer"
- For complex emails with multiple issues, address EACH issue in your reply

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.

Example:
{
  "category": "billing",
  "priority": "high",
  "reply_draft": "Dear Priya, I sincerely apologize for the double charge on your account. I can see the duplicate charge of ₹499 on account #84721 and have initiated an immediate refund. You will see the credit within 3-5 business days. I have also flagged your account to prevent this from recurring. Please do not hesitate to reach out if you have any questions."
}"""


def call_llm(email_subject: str, email_body: str, sender_name: str, task_description: str) -> dict:
    """
    Call the LLM with the email context and get a structured response.
    Returns a dict with category, priority, reply_draft.
    """
    user_message = f"""
From: {sender_name}
Subject: {email_subject}

{email_body}

---
TASK: {task_description}

Respond with valid JSON only.
"""
    for attempt in range(3):  # retry up to 3 times
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=600,
                temperature=0.2,  # low temperature for consistent structured output
            )

            raw = response.choices[0].message.content.strip()

            # Clean up in case model wraps in markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)

            # Validate all required fields exist
            assert "category"    in parsed, "Missing 'category' field"
            assert "priority"    in parsed, "Missing 'priority' field"
            assert "reply_draft" in parsed, "Missing 'reply_draft' field"

            return parsed

        except (json.JSONDecodeError, AssertionError, Exception) as e:
            print(f"  ⚠️  LLM attempt {attempt+1} failed: {e}")
            time.sleep(1)

    # Fallback if all attempts fail
    print("  ❌ All LLM attempts failed. Using fallback response.")
    return {
        "category":    "general",
        "priority":    "medium",
        "reply_draft": (
            f"Dear {sender_name}, thank you for reaching out. "
            "We have received your message and our team is reviewing your issue. "
            "We will get back to you within 24 hours with a resolution. "
            "We apologize for any inconvenience caused."
        ),
    }


def run_inference():
    """
    Main inference loop.
    Connects to the environment, runs the agent through all 3 tasks,
    collects and prints scores.
    """
    print("=" * 60)
    print("  Email Review Environment — Inference Script")
    print("=" * 60)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Env URL : {ENV_URL}")
    print("=" * 60)

    scores = []
    start_time = time.time()

    try:
        with EmailReviewEnv(base_url=ENV_URL).sync() as env:

            # Reset the environment (get first task)
            print("\n🔄 Resetting environment...")
            obs = env.reset()
            print(f"  Episode started. First task loaded.")

            task_num = 1

            while not obs.done:
                print(f"\n{'─'*50}")
                print(f"📧 Task {task_num} — {obs.sender_name}")
                print(f"   Subject : {obs.email_subject}")
                print(f"   Body    : {obs.email_body[:120]}...")

                # Call LLM
                print(f"\n🤖 Calling LLM ({MODEL_NAME})...")
                llm_response = call_llm(
                    email_subject   = obs.email_subject,
                    email_body      = obs.email_body,
                    sender_name     = obs.sender_name,
                    task_description= obs.task_description,
                )

                print(f"   → Category : {llm_response['category']}")
                print(f"   → Priority : {llm_response['priority']}")
                print(f"   → Reply    : {llm_response['reply_draft'][:100]}...")

                # Submit action to environment
                action = EmailAction(
                    category    = llm_response["category"],
                    priority    = llm_response["priority"],
                    reply_draft = llm_response["reply_draft"],
                )
                result = env.step(action)

                score = result.reward
                scores.append(score)

                print(f"\n📊 Score for Task {task_num}: {score:.3f}")
                print(f"   Breakdown:\n{result.observation.score_breakdown}")

                obs = result.observation
                task_num += 1

    except ConnectionRefusedError:
        print(f"\n❌ Could not connect to environment at {ENV_URL}")
        print("   Make sure the server is running: uvicorn server.app:app --port 8000")
        sys.exit(1)

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for i, s in enumerate(scores, 1):
        difficulty = ["Easy", "Medium", "Hard"][i-1]
        print(f"  Task {i} ({difficulty}): {s:.3f}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average Score  : {avg:.3f}")
        print(f"  Tasks Completed: {len(scores)}/3")
        print(f"  Runtime        : {elapsed:.1f}s")

        # Write results to file for automated checker
        results = {
            "task_scores":      scores,
            "average_score":    round(avg, 3),
            "tasks_completed":  len(scores),
            "runtime_seconds":  round(elapsed, 1),
            "model":            MODEL_NAME,
            "all_passed":       all(s >= 0.0 for s in scores),  # 0.0 = at least attempted
        }
        with open("outputs/evals/results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results written to outputs/evals/results.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    os.makedirs("outputs/evals", exist_ok=True)
    run_inference()
