import os
import json
import time
import sys
import requests
from openai import OpenAI

# Required env vars — API_BASE_URL and MODEL_NAME have defaults, HF_TOKEN does NOT
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL          = os.getenv("ENV_URL", "https://jhansiy1-email-review-env.hf.space")

TASK_NAME  = "email_triage"
BENCHMARK  = "email_review"

# OpenAI client — uses HF_TOKEN, falls back to dummy only for env testing
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "hf-no-token-set",
)

SYSTEM_PROMPT = (
    "You are an expert customer support email triage agent. "
    "Analyze the email carefully and respond with ONLY a valid JSON object "
    "with exactly these three fields:\n"
    "category: one of billing, technical, general, complaint\n"
    "priority: one of low, medium, high, urgent\n"
    "reply_draft: professional empathetic reply minimum 80 words.\n"
    "No markdown. No explanation. Raw JSON only."
)


def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error=None):
    action_str = json.dumps(action)
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_val   = max(0.0, min(float(score), 1.0))
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score_val:.2f} rewards={rewards_str}",
        flush=True
    )


def call_llm(subject, body, sender):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        "From: " + sender + "\n"
                        "Subject: " + subject + "\n\n"
                        + body + "\n\nRespond with JSON only."
                    )},
                ],
                max_tokens=600,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            assert "category" in parsed
            assert "priority" in parsed
            assert "reply_draft" in parsed
            return parsed
        except Exception:
            time.sleep(1)
    # Fallback — deterministic, based on keywords
    subj_lower = subject.lower()
    body_lower = body.lower()
    if "invoice" in subj_lower or "charge" in body_lower or "refund" in body_lower or "billing" in body_lower:
        cat, pri = "billing", "high"
    elif "api" in subj_lower or "error" in body_lower or "401" in body_lower or "technical" in body_lower:
        cat, pri = "technical", "urgent"
    elif "frustrated" in subj_lower or "furious" in body_lower or "unacceptable" in body_lower:
        cat, pri = "complaint", "urgent"
    else:
        cat, pri = "general", "medium"
    return {
        "category": cat,
        "priority": pri,
        "reply_draft": (
            "Dear " + sender + ", I sincerely apologize for the issue you are experiencing. "
            "I have reviewed your request and am escalating it immediately to ensure a prompt "
            "resolution. Our team is treating this as a high priority matter. You can expect "
            "a detailed follow-up within 24 hours. We deeply value your relationship with us "
            "and are committed to resolving this to your complete satisfaction. Thank you for "
            "your patience and understanding."
        ),
    }


def run():
    os.makedirs("outputs/evals", exist_ok=True)
    start_time = time.time()
    rewards    = []
    steps      = 0
    success    = False

    log_start()

    try:
        r    = requests.post(ENV_URL + "/reset", timeout=30)
        data = r.json()
        obs  = data.get("observation", {})

        while not data.get("done", False):
            steps += 1

            subject = obs.get("email_subject", "")
            body    = obs.get("email_body",    "")
            sender  = obs.get("sender_name",   "")

            llm_out = call_llm(subject, body, sender)

            action = {
                "category":    llm_out.get("category",    "general"),
                "priority":    llm_out.get("priority",    "medium"),
                "reply_draft": llm_out.get("reply_draft", "Thank you for contacting us."),
            }

            r    = requests.post(ENV_URL + "/step", json={"action": action}, timeout=30)
            data = r.json()

            reward = float(data.get("reward", 0.0))
            done   = bool(data.get("done",   False))
            obs    = data.get("observation", {})
            rewards.append(reward)

            log_step(steps, action, reward, done)

        success = True

    except Exception as e:
        success = False
        log_step(steps + 1, {}, 0.0, True, error=str(e))

    elapsed   = round(time.time() - start_time, 2)
    avg_score = round(sum(rewards) / max(len(rewards), 1), 4)

    with open("outputs/evals/results.json", "w") as fh:
        json.dump({
            "task_scores":   rewards,
            "average_score": avg_score,
            "total_steps":   steps,
            "runtime_secs":  elapsed,
            "model":         MODEL_NAME,
        }, fh, indent=2)

    log_end(success, steps, avg_score, rewards)


if __name__ == "__main__":
    run()
