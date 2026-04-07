"""
test_local.py  -  Run before deploying to verify everything works.

Usage:   python test_local.py
All tests should show PASS.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import EmailAction
from server.environment import EmailReviewEnvironment, TASKS, grade_action

_failed = False

def check(cond, label):
    global _failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}]  {label}")
    if not cond:
        _failed = True

def section(title):
    print(f"\n--- {title} ---")


section("1. Model creation")
a = EmailAction(category="billing", priority="high",
                reply_draft="Dear customer, I apologize and will process your refund immediately.")
check(a.category == "billing", "category set")
check(a.priority == "high", "priority set")
check(len(a.reply_draft) > 10, "reply_draft not empty")


section("2. Grader — perfect answer")
task1 = TASKS[0]
perfect = EmailAction(
    category="billing", priority="high",
    reply_draft=(
        "Dear Priya, I sincerely apologize for the double charge on account #84721. "
        "I can confirm a full refund has been initiated for the duplicate billing. "
        "Please allow 3-5 business days. We apologize and will escalate if needed."
    )
)
score, breakdown = grade_action(task1, perfect)
check(score >= 0.85, f"Perfect answer scores >= 0.85 (got {score})")
print(f"  Score: {score}")


section("3. Grader — wrong answer scores low")
wrong = EmailAction(category="general", priority="low", reply_draft="Thanks.")
score_w, _ = grade_action(task1, wrong)
check(score_w <= 0.15, f"Wrong answer scores <= 0.15 (got {score_w})")


section("4. Grader — partial credit works")
partial = EmailAction(
    category="billing", priority="low",  # correct cat, wrong priority
    reply_draft="Dear Priya, I apologize for the double charge. We will process a refund and escalate to billing."
)
score_p, _ = grade_action(task1, partial)
check(0.20 < score_p < 0.85, f"Partial answer is between 0.20-0.85 (got {score_p})")


section("5. Full episode — 3 tasks, no server needed")
env = EmailReviewEnvironment()
obs = env.reset()
check(obs.email_subject == TASKS[0]["email_subject"], "reset() shows task 1")
check(not obs.done, "reset() done=False")

r1 = env.step(EmailAction(
    category="billing", priority="high",
    reply_draft="Dear Priya, apologies for the double charge on your account. Refund will be processed immediately. We will escalate and ensure this doesn't happen again."
))
check(r1.reward > 0.3, f"Task 1 has reward > 0.3 (got {r1.reward})")
check(not r1.observation.done, "Task 1 not yet done")

r2 = env.step(EmailAction(
    category="complaint", priority="urgent",
    reply_draft=(
        "Dear Ravi, I sincerely apologize for the 3-day outage affecting your business. "
        "As a premium subscriber this is unacceptable. I am personally escalating to our "
        "senior team right now and we will resolve this today. Full compensation will be reviewed."
    )
))
check(r2.reward > 0.3, f"Task 2 has reward > 0.3 (got {r2.reward})")
check(not r2.observation.done, "Task 2 not yet done")

r3 = env.step(EmailAction(
    category="technical", priority="urgent",
    reply_draft=(
        "Dear Ananya, I apologize for both issues affecting your team. "
        "For the API 401 error: there is a known propagation delay after key rotation. "
        "Please regenerate the api key and wait 2 hours. We will escalate to auth engineers. "
        "For billing: I can see account ENT-00291 is on the wrong tier. I am escalating "
        "to enterprise billing to correct this to the Enterprise plan immediately. "
        "Your SLA and dedicated support will be restored within 4 hours."
    )
))
check(r3.observation.done, "Task 3 marks episode done=True")
check(r3.reward > 0.3, f"Task 3 has reward > 0.3 (got {r3.reward})")

obs2 = env.reset()
check(obs2.email_subject == TASKS[0]["email_subject"], "Second reset() works fine")


section("6. Task structure validation")
required_fields = ["id","difficulty","sender_name","email_subject","email_body",
                   "task_description","correct_category","correct_priority",
                   "required_keywords","forbidden_phrases","min_reply_length"]
check(len(TASKS) >= 3, f"At least 3 tasks defined (got {len(TASKS)})")
for i, task in enumerate(TASKS):
    for f in required_fields:
        check(f in task, f"Task {i+1} has field '{f}'")
difficulties = [t["difficulty"] for t in TASKS]
for d in ["easy", "medium", "hard"]:
    check(d in difficulties, f"Has a '{d}' task")


section("7. All scores in 0.0-1.0 range (random stress test)")
for task in TASKS:
    for _ in range(10):
        action = EmailAction(
            category=random.choice(["billing","technical","general","complaint"]),
            priority=random.choice(["low","medium","high","urgent"]),
            reply_draft="word " * random.randint(1, 150)
        )
        s, _ = grade_action(task, action)
        check(0.0 <= s <= 1.0, f"Score {s:.3f} in [0,1] for task '{task['id']}'")


print(f"\n{'='*50}")
if _failed:
    print("  SOME TESTS FAILED — fix before submitting.")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED — safe to deploy!")
print('='*50)
