"""
inference.py — Baseline inference script for Customer Support Ticket Triage OpenEnv.

Environment variables:
  API_BASE_URL  — LLM API base URL
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key
"""

import os
import json
import time
import sys

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

try:
    from environment import TicketTriageEnv, Action
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import TicketTriageEnv, Action

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("[START]")
    print(json.dumps({"task_id": "none", "error": "HF_TOKEN not set"}))
    print("[END]")
    print(json.dumps({"task_id": "none", "score": 0.0, "total_reward": 0.0}))
    sys.stdout.flush()
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = {"easy": 5, "medium": 8, "hard": 12}

SYSTEM_PROMPT = """You are an expert customer support agent. You receive a support ticket and must triage it correctly.

For each ticket you can take ONE of these actions (respond with valid JSON only):

1. Categorize:
   {"action_type": "categorize", "category": "<billing|technical|account|feature_request|spam>"}

2. Prioritize:
   {"action_type": "prioritize", "priority": "<low|medium|high|urgent>"}

3. Respond to customer:
   {"action_type": "respond", "response_text": "<your response>"}

4. Escalate:
   {"action_type": "escalate", "escalate_to": "<tier2|billing_team|engineering>"}

5. Close ticket:
   {"action_type": "close", "close_reason": "<resolved|spam|duplicate>"}

Strategy:
- First categorize, then prioritize, then decide on response/escalation/close.
- Enterprise or urgent issues escalate to the right team.
- Spam close immediately.
- Billing disputes escalate to billing_team.
- API/security issues escalate to engineering.
- Feature requests close with resolved after acknowledging.
- Always respond first to high-stakes tickets before escalating.

Return ONLY valid JSON with no explanation or markdown fences."""


def build_user_prompt(obs: dict) -> str:
    return f"""Ticket ID: {obs['ticket_id']}
Subject: {obs['subject']}
From: {obs['sender_email']} (Tier: {obs['sender_tier']})
Message:
{obs['body']}

Step {obs['step_number']} — What is your next action? Return JSON only."""


def call_llm(messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM_ERROR: {str(e)}", file=sys.stderr)
        return '{"action_type": "categorize", "category": "technical"}'


def parse_action(raw: str) -> Action:
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        data = json.loads(clean)
        return Action(**data)
    except Exception:
        return Action(action_type="categorize", category="technical")


def run_task(task_id: str) -> dict:
    try:
        env = TicketTriageEnv(task_id)
        obs = env.reset()
        obs_dict = obs.model_dump()
    except Exception as e:
        print("[START]")
        print(json.dumps({"task_id": task_id, "error": str(e)}))
        print("[END]")
        print(json.dumps({"task_id": task_id, "score": 0.0, "total_reward": 0.0}))
        sys.stdout.flush()
        return {"task_id": task_id, "steps": [], "total_reward": 0.0, "final_score": 0.0}

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    task_result = {
        "task_id": task_id,
        "steps": [],
        "total_reward": 0.0,
        "final_score": 0.0,
    }

    # ── [START] block ──────────────────────────────────────────────────────────
    print("[START]")
    print(json.dumps({
        "task_id": task_id,
        "ticket_id": obs_dict["ticket_id"],
        "subject": obs_dict["subject"],
        "sender_tier": obs_dict["sender_tier"],
        "model": MODEL_NAME,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

    for step_num in range(1, MAX_STEPS[task_id] + 1):
        try:
            user_msg = build_user_prompt(obs_dict)
            conversation.append({"role": "user", "content": user_msg})

            raw_action = call_llm(conversation)
            conversation.append({"role": "assistant", "content": raw_action})

            action = parse_action(raw_action)
            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump()

            # ── [STEP] block ───────────────────────────────────────────────────
            print("[STEP]")
            print(json.dumps({
                "task_id": task_id,
                "step": step_num,
                "action": action.model_dump(),
                "reward": reward.value,
                "reward_breakdown": reward.breakdown,
                "reward_message": reward.message,
                "done": done,
                "cumulative_reward": info["cumulative_reward"],
                "timestamp": time.time(),
            }))
            sys.stdout.flush()

            task_result["steps"].append({
                "step": step_num,
                "action": action.model_dump(),
                "reward": reward.value,
                "done": done,
            })
            task_result["total_reward"] += reward.value

            if done:
                break

        except Exception as e:
            print("[STEP]")
            print(json.dumps({
                "task_id": task_id,
                "step": step_num,
                "error": str(e),
                "reward": 0.0,
                "done": True,
                "timestamp": time.time(),
            }))
            sys.stdout.flush()
            break

    try:
        final_score = env.grade()
    except Exception:
        final_score = 0.0

    task_result["final_score"] = final_score

    # ── [END] block ────────────────────────────────────────────────────────────
    print("[END]")
    print(json.dumps({
        "task_id": task_id,
        "total_steps": len(task_result["steps"]),
        "total_reward": round(task_result["total_reward"], 4),
        "final_score": final_score,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

    return task_result


def main():
    all_results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id)
            all_results.append(result)
        except Exception as e:
            print("[START]")
            print(json.dumps({"task_id": task_id, "error": str(e)}))
            print("[END]")
            print(json.dumps({"task_id": task_id, "score": 0.0, "total_reward": 0.0}))
            sys.stdout.flush()
            all_results.append({"task_id": task_id, "final_score": 0.0})
        time.sleep(1)

    avg_score = sum(r["final_score"] for r in all_results) / len(all_results)
    print(json.dumps({
        "event": "SUMMARY",
        "tasks": {r["task_id"]: r["final_score"] for r in all_results},
        "average_score": round(avg_score, 4),
        "timestamp": time.time(),
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
