"""
inference.py — Email Review Environment Baseline
Follows the exact required log format: [START] [STEP] [END]
Uses OpenAI client for all LLM calls.
"""
import os, json, time, sys, requests
from openai import OpenAI

# ── Checklist item 2 + 3: API_BASE_URL and MODEL_NAME have defaults,
# HF_TOKEN has NO default (must be set by user)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

# ── Checklist item 4: OpenAI client configured via the above variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key",
)

SYSTEM_PROMPT = """You are an expert customer support email triage agent.
For each customer email respond with ONLY a valid JSON object with these exact fields:
  "category": one of billing, technical, general, complaint
  "priority": one of low, medium, high, urgent
  "reply_draft": professional empathetic reply minimum 80 words
No markdown. No explanation. Just the raw JSON object."""


def call_llm(subject, body, sender):
    """Call LLM using OpenAI client — required by contest rules."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "From: " + sender + "\nSubject: " + subject + "\n\n" + body + "\n\nJSON only:"},
                ],
                max_tokens=600,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            time.sleep(1)
    return {
        "category": "general",
        "priority": "medium",
        "reply_draft": (
            "Dear " + sender + ", thank you for reaching out to us. "
            "We have received your message and our team is reviewing your issue urgently. "
            "We sincerely apologize for any inconvenience this has caused you. "
            "Our team will contact you within 24 hours with a full resolution. "
            "Please do not hesitate to reach out if you need any further assistance."
        )
    }


def run():
    os.makedirs("outputs/evals", exist_ok=True)
    start_time = time.time()

    # ── Checklist item 5: [START] log — exact required format
    log_start = {
        "type":    "START",
        "env_url": ENV_URL,
        "model":   MODEL_NAME,
    }
    print(json.dumps(log_start))
    sys.stdout.flush()

    # Reset environment
    r = requests.post(ENV_URL + "/reset", timeout=30)
    data = r.json()
    obs  = data.get("observation", {})

    step_num   = 0
    all_scores = []

    while not data.get("done", False):
        step_num += 1
        subject = obs.get("email_subject", "")
        body    = obs.get("email_body",    "")
        sender  = obs.get("sender_name",   "")

        llm_out = call_llm(subject, body, sender)

        action = {
            "category":    llm_out.get("category",    "general"),
            "priority":    llm_out.get("priority",    "medium"),
            "reply_draft": llm_out.get("reply_draft", "Thank you for contacting us. We will resolve your issue promptly and apologize for any inconvenience caused to you."),
        }

        r = requests.post(ENV_URL + "/step", json={"action": action}, timeout=30)
        data   = r.json()
        reward = float(data.get("reward", 0.0))
        done   = bool(data.get("done", False))
        obs    = data.get("observation", {})
        all_scores.append(reward)

        # ── Checklist item 5: [STEP] log — exact required format
        log_step = {
            "type":   "STEP",
            "step":   step_num,
            "action": action,
            "reward": reward,
            "done":   done,
        }
        print(json.dumps(log_step))
        sys.stdout.flush()

    elapsed   = round(time.time() - start_time, 2)
    avg_score = round(sum(all_scores) / max(len(all_scores), 1), 4)

    results = {
        "task_scores":   all_scores,
        "average_score": avg_score,
        "total_steps":   step_num,
        "runtime_secs":  elapsed,
        "model":         MODEL_NAME,
    }
    with open("outputs/evals/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Checklist item 5: [END] log — exact required format
    log_end = {
        "type":          "END",
        "total_steps":   step_num,
        "average_score": avg_score,
        "runtime":       elapsed,
        "scores":        all_scores,
    }
    print(json.dumps(log_end))
    sys.stdout.flush()


if __name__ == "__main__":
    run()