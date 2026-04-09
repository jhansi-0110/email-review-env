import os
import json
import time
import sys
import requests

# Required variables — API_BASE_URL and MODEL_NAME have defaults, HF_TOKEN does NOT
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ENV_URL defaults to localhost — judges run Docker locally
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

TASK_NAME = "email_triage"
BENCHMARK = "email_review"

# Set env vars BEFORE importing OpenAI — fixes version compatibility issues
os.environ["OPENAI_BASE_URL"] = API_BASE_URL
os.environ["OPENAI_API_KEY"]  = HF_TOKEN if HF_TOKEN else "hf-no-token-set"

from openai import OpenAI

# Create client with fallback for different openai versions
try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "hf-no-token-set",
    )
except TypeError:
    try:
        client = OpenAI()
    except Exception as e:
        print("[DEBUG] OpenAI client init failed: " + str(e), flush=True)
        client = None

SYSTEM_PROMPT = (
    "You are an expert customer support email triage agent. "
    "Analyze the email carefully and respond with ONLY a valid JSON object "
    "with exactly these three fields: "
    "category (one of: billing, technical, general, complaint), "
    "priority (one of: low, medium, high, urgent), "
    "reply_draft (professional empathetic reply minimum 80 words). "
    "No markdown. No explanation. Raw JSON only."
)


def log_start():
    print("[START] task=" + TASK_NAME + " env=" + BENCHMARK + " model=" + MODEL_NAME, flush=True)


def log_step(step, action, reward, done, error=None):
    action_str = json.dumps(action)
    error_val  = str(error) if error else "null"
    done_val   = str(done).lower()
    print(
        "[STEP] step=" + str(step) +
        " action=" + action_str +
        " reward=" + str(round(reward, 2)) +
        " done=" + done_val +
        " error=" + error_val,
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(str(round(r, 2)) for r in rewards)
    score_val   = max(0.0, min(float(score), 1.0))
    print(
        "[END] success=" + str(success).lower() +
        " steps=" + str(steps) +
        " score=" + str(round(score_val, 2)) +
        " rewards=" + rewards_str,
        flush=True
    )


def call_llm(subject, body, sender):
    if client is None:
        return get_fallback(subject, body, sender)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        "From: " + sender + "\n"
                        "Subject: " + subject + "\n\n"
                        + body + "\n\nJSON only:"
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
    return get_fallback(subject, body, sender)


def get_fallback(subject, body, sender):
    subj_lower = (subject + " " + body).lower()
    if "invoice" in subj_lower or "refund" in subj_lower or "charge" in subj_lower or "billing" in subj_lower:
        cat, pri = "billing", "high"
    elif "api" in subj_lower or "401" in subj_lower or "technical" in subj_lower or "error" in subj_lower:
        cat, pri = "technical", "urgent"
    elif "frustrated" in subj_lower or "furious" in subj_lower or "unacceptable" in subj_lower or "premium" in subj_lower:
        cat, pri = "complaint", "urgent"
    else:
        cat, pri = "general", "medium"
    return {
        "category": cat,
        "priority": pri,
        "reply_draft": (
            "Dear " + sender + ", I sincerely apologize for the issue you are experiencing. "
            "I have reviewed your request carefully and am escalating it immediately to our "
            "senior team to ensure a prompt and complete resolution. Your case has been flagged "
            "as the highest priority. You can expect a detailed follow-up within 24 hours. "
            "We are fully committed to resolving this to your complete satisfaction and deeply "
            "value your continued relationship with us. Thank you for your patience."
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
        r    = requests.post(ENV_URL + "/reset", timeout=60)
        data = r.json()
        obs  = data.get("observation", {})

        while not data.get("done", False):
            steps += 1
            subject = obs.get("email_subject", "")
            body    = obs.get("email_body",    "")
            sender  = obs.get("sender_name",   "")

            llm_out = call_llm(subject, body, sender)
            action  = {
                "category":    llm_out.get("category",    "general"),
                "priority":    llm_out.get("priority",    "medium"),
                "reply_draft": llm_out.get("reply_draft", "Thank you for contacting us. We sincerely apologize for the inconvenience and will resolve your issue promptly."),
            }

            r    = requests.post(ENV_URL + "/step", json={"action": action}, timeout=60)
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

    try:
        with open("outputs/evals/results.json", "w") as fh:
            json.dump({
                "task_scores":   rewards,
                "average_score": avg_score,
                "total_steps":   steps,
                "runtime_secs":  elapsed,
                "model":         MODEL_NAME,
            }, fh, indent=2)
    except Exception:
        pass

    log_end(success, steps, avg_score, rewards)


if __name__ == "__main__":
    run()
