"""
validate.py — Pre-submission validation script.
Run this before submitting to catch any issues early.
"""

import sys
import json
import yaml
import importlib

PASS = "✅"
FAIL = "❌"
results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        results.append((name, False, str(e)))


# 1. openenv.yaml exists and is valid
def check_yaml():
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert "name" in data
    assert "tasks" in data
    assert len(data["tasks"]) >= 3, "Need at least 3 tasks"

check("openenv.yaml valid with 3+ tasks", check_yaml)


# 2. Environment imports and models are typed
def check_models():
    from environment import Observation, Action, Reward, TicketTriageEnv
    obs = Observation(
        ticket_id="T1", subject="Test", body="Test body",
        sender_email="a@b.com", sender_tier="free",
        created_at="2025-01-01", queue_size=5,
        task_id="easy", step_number=0
    )
    assert obs.ticket_id == "T1"
    act = Action(action_type="categorize", category="billing")
    assert act.action_type == "categorize"
    rwd = Reward(value=0.75, breakdown={"category": 0.75}, message="ok")
    assert 0.0 <= rwd.value <= 1.0

check("Pydantic models (Observation, Action, Reward) valid", check_models)


# 3. reset() returns Observation
def check_reset():
    from environment import TicketTriageEnv
    for task_id in ["easy", "medium", "hard"]:
        env = TicketTriageEnv(task_id)
        obs = env.reset()
        assert obs.task_id == task_id
        assert obs.step_number == 0
        assert not obs.done

check("reset() works for all 3 tasks", check_reset)


# 4. step() returns correct tuple
def check_step():
    from environment import TicketTriageEnv, Action
    env = TicketTriageEnv("easy")
    env.reset()
    action = Action(action_type="categorize", category="account")
    obs, reward, done, info = env.step(action)
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(done, bool)
    assert "step" in info

check("step() returns (obs, reward, done, info)", check_step)


# 5. state() returns dict
def check_state():
    from environment import TicketTriageEnv
    env = TicketTriageEnv("medium")
    env.reset()
    s = env.state()
    assert isinstance(s, dict)
    assert "task_id" in s

check("state() returns valid dict", check_state)


# 6. grade() returns float 0–1
def check_grade():
    from environment import TicketTriageEnv, Action
    for task_id in ["easy", "medium", "hard"]:
        env = TicketTriageEnv(task_id)
        env.reset()
        env.step(Action(action_type="categorize", category="technical"))
        env.step(Action(action_type="prioritize", priority="high"))
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for {task_id}"

check("grade() returns float in [0.0, 1.0] for all tasks", check_grade)


# 7. Rewards are shaped (not constant)
def check_shaped_rewards():
    from environment import TicketTriageEnv, Action
    env = TicketTriageEnv("easy")
    env.reset()
    rewards = []
    actions = [
        Action(action_type="categorize", category="billing"),
        Action(action_type="prioritize", priority="urgent"),
        Action(action_type="close", close_reason="resolved"),
    ]
    for a in actions:
        _, r, done, _ = env.step(a)
        rewards.append(r.value)
        if done:
            break
    # Not all same value → shaped
    assert len(set(rewards)) > 1 or len(rewards) == 1, "Rewards appear constant"

check("Reward function provides varying signal (shaped)", check_shaped_rewards)


# 8. inference.py exists
def check_inference():
    import os
    assert os.path.exists("inference.py"), "inference.py not found"
    with open("inference.py") as f:
        content = f.read()
    assert "[START]" in content or '"START"' in content, "Missing START log event"
    assert "[STEP]" in content or '"STEP"' in content, "Missing STEP log event"
    assert "[END]" in content or '"END"' in content, "Missing END log event"
    assert "OpenAI" in content or "openai" in content, "Must use OpenAI client"

check("inference.py exists with correct logging format", check_inference)


# 9. Dockerfile exists
def check_dockerfile():
    import os
    assert os.path.exists("Dockerfile")
    with open("Dockerfile") as f:
        content = f.read()
    assert "7860" in content, "Must expose port 7860 for HF Spaces"

check("Dockerfile exists and exposes port 7860", check_dockerfile)


# Summary
print("\n" + "="*50)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"Results: {passed}/{total} checks passed")

if passed < total:
    print("\nFailed checks:")
    for name, ok, err in results:
        if not ok:
            print(f"  ❌ {name}: {err}")
    sys.exit(1)
else:
    print("🎉 All checks passed! Ready to submit.")
