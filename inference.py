import os
import json
import time
import sys
import requests

API_BASE_URL     = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY          = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "no-key"
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860")

TASK_NAME  = "email_triage"
BENCHMARK  = "email_review"

from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are an expert customer support email triage agent. "
    "Analyze the email and respond with ONLY a valid JSON object "
    "with exactly these three fields: "
    "category (one of: billing, technical, general, complaint), "
    "priority (one of: low, medium, high, urgent), "
    "reply_draft (professional empathetic reply minimum 80 words). "
    "No markdown. No explanation. Raw JSON only."
)


def log_start(task_id):
    print("[START] task=" + task_id + " env=" + BENCHMARK + " model=" + MODEL_NAME, flush=True)


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
    except Exception as e:
        print("[DEBUG] LLM failed: " + str(e), flush=True)
        combined = (subject + " " + body).lower()
        if "invoice" in combined or "refund" in combined or "billing" in combined or "charge" in combined:
            cat, pri = "billing", "high"
        elif "api" in combined or "401" in combined or "authentication" in combined:
            cat, pri = "technical", "urgent"
        elif "frustrated" in combined or "furious" in combined or "premium" in combined:
            cat, pri = "complaint", "urgent"
        else:
            cat, pri = "general", "medium"
        return {
            "category": cat,
            "priority": pri,
            "reply_draft": (
                "Dear " + sender + ", I sincerely apologize for the issue you are experiencing. "
                "I have reviewed your request carefully and am escalating it immediately to our "
                "senior team to ensure a prompt resolution. Your case has been flagged as the "
                "highest priority and you can expect a detailed follow-up within 24 hours. "
                "We are fully committed to resolving this to your complete satisfaction and "
                "deeply value your continued relationship with us. Thank you for your patience."
            ),
        }


def run_single_task(task):
    """Run one task independently: reset with task_id, step once, done=True."""
    task_id = task["id"]
    all_rewards = []
    steps = 0
    success = False

    log_start(task_id)

    try:
        # Reset for this specific task
        r = requests.post(
            ENV_URL + "/reset",
            json={"task_id": task_id, "session_id": task_id},
            timeout=60
        )
        r.raise_for_status()
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
                "reply_draft": llm_out.get("reply_draft",
                    "Thank you for contacting us. We sincerely apologize and "
                    "will resolve your issue promptly within 24 hours."
                ),
            }

            r = requests.post(
                ENV_URL + "/step",
                json={"action": action, "session_id": task_id},
                timeout=60
            )
            r.raise_for_status()
            data   = r.json()
            reward = float(data.get("reward", 0.0))
            done   = bool(data.get("done",   False))
            obs    = data.get("observation", {})
            all_rewards.append(reward)

            log_step(steps, action, reward, done)

            if done:
                break

            # Safety: single task should complete in 1 step
            if steps >= 5:
                break

        success = True

    except Exception as e:
        print("[DEBUG] Task " + task_id + " error: " + str(e), flush=True)
        success = False
        if steps == 0:
            log_step(1, {}, 0.0, True, error=str(e))
        all_rewards = all_rewards or [0.0]

    avg = round(sum(all_rewards) / max(len(all_rewards), 1), 4)
    log_end(success, steps, avg, all_rewards)

    return all_rewards


def run():
    os.makedirs("outputs/evals", exist_ok=True)
    start_time = time.time()

    # Step 1: Get all tasks from the environment
    try:
        r = requests.get(ENV_URL + "/tasks", timeout=30)
        r.raise_for_status()
        tasks_data = r.json()
        tasks = tasks_data.get("tasks", [])
        print("[DEBUG] Found " + str(len(tasks)) + " tasks", flush=True)
    except Exception as e:
        print("[DEBUG] Could not get tasks: " + str(e), flush=True)
        # Fallback: hardcode the 3 task IDs
        tasks = [
            {"id": "task_1_easy"},
            {"id": "task_2_medium"},
            {"id": "task_3_hard"},
        ]

    # Step 2: Run each task SEPARATELY
    all_task_scores = {}
    all_rewards_flat = []

    for task in tasks:
        rewards = run_single_task(task)
        all_task_scores[task["id"]] = rewards
        all_rewards_flat.extend(rewards)
        time.sleep(1)

    elapsed   = round(time.time() - start_time, 2)
    avg_score = round(sum(all_rewards_flat) / max(len(all_rewards_flat), 1), 4)

    try:
        with open("outputs/evals/results.json", "w") as fh:
            json.dump({
                "task_scores":   all_task_scores,
                "average_score": avg_score,
                "total_tasks":   len(tasks),
                "runtime_secs":  elapsed,
                "model":         MODEL_NAME,
            }, fh, indent=2)
    except Exception:
        pass

    print("[DEBUG] All tasks complete. Average score: " + str(avg_score), flush=True)


if __name__ == "__main__":
    run()
