"""
inference.py — MUST be at root, MUST follow exact [START][STEP][END] log format
"""
import os, json, time, sys
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key-needed",
)

SYSTEM_PROMPT = """You are an expert customer support email triage agent.
For each email respond with ONLY a JSON object with exactly these fields:
  "category": one of billing, technical, general, complaint
  "priority": one of low, medium, high, urgent
  "reply_draft": professional reply (minimum 50 words)
No markdown, no explanation, just the JSON."""

def call_llm(subject, body, sender):
    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"From: {sender}\nSubject: {subject}\n\n{body}\n\nRespond with JSON only."}
                ],
                max_tokens=500,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            time.sleep(1)
    return {"category": "general", "priority": "medium",
            "reply_draft": f"Dear {sender}, thank you for contacting us. We have received your message and will respond within 24 hours. We apologize for any inconvenience caused and will work to resolve your issue as quickly as possible."}

def run():
    import requests

    # [START] log — REQUIRED exact format
    print(json.dumps({"type": "START", "env_url": ENV_URL, "model": MODEL_NAME}))
    sys.stdout.flush()

    start_time = time.time()
    os.makedirs("outputs/evals", exist_ok=True)

    # Reset
    r = requests.post(f"{ENV_URL}/reset", timeout=30)
    data = r.json()
    obs = data.get("observation", {})

    step_num = 0
    all_scores = []

    while not data.get("done", False):
        step_num += 1
        subject = obs.get("email_subject", "")
        body    = obs.get("email_body", "")
        sender  = obs.get("sender_name", "")

        # Call LLM
        llm_out = call_llm(subject, body, sender)

        action = {
            "category":    llm_out.get("category", "general"),
            "priority":    llm_out.get("priority", "medium"),
            "reply_draft": llm_out.get("reply_draft", "Thank you for contacting us. We will resolve your issue shortly.")
        }

        # Step
        r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
        data = r.json()
        reward = data.get("reward", 0.0)
        all_scores.append(reward)
        obs = data.get("observation", {})

        # [STEP] log — REQUIRED exact format
        print(json.dumps({
            "type":   "STEP",
            "step":   step_num,
            "action": action,
            "reward": reward,
            "done":   data.get("done", False)
        }))
        sys.stdout.flush()

    elapsed = round(time.time() - start_time, 2)
    avg_score = round(sum(all_scores) / max(len(all_scores), 1), 4)

    results = {
        "task_scores":   all_scores,
        "average_score": avg_score,
        "total_steps":   step_num,
        "runtime":       elapsed,
        "model":         MODEL_NAME,
    }

    with open("outputs/evals/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # [END] log — REQUIRED exact format
    print(json.dumps({
        "type":          "END",
        "total_steps":   step_num,
        "average_score": avg_score,
        "runtime":       elapsed,
        "scores":        all_scores,
    }))
    sys.stdout.flush()

if __name__ == "__main__":
    run()
