---
title: Customer Support  Ticket Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - agent-evaluation
---

# 🎫 Customer Support Ticket Triage — OpenEnv

An **OpenEnv-compliant** reinforcement learning environment where AI agents learn to triage customer support tickets — a real-world task performed by thousands of support teams daily.

---

## 🌍 Motivation

Customer support triage is a critical, high-volume task: agents must correctly categorize issues, set priority, route to the right team, and sometimes draft responses — all under time pressure. This environment lets you train and evaluate agents on this workflow with graded, reproducible scoring.

---

## 🗂 Environment Overview

| Property | Value |
|---|---|
| **Domain** | Customer Support / NLP |
| **Task type** | Sequential decision-making over ticket triage |
| **Action space** | Discrete + text (5 action types) |
| **Observation** | Structured ticket + context |
| **Reward** | Shaped per-step (0.0–1.0) |
| **Tasks** | 3 (easy → medium → hard) |

---

## 📥 Observation Space

Each step the agent receives:

```json
{
  "ticket_id": "TKT-1001",
  "subject": "How do I reset my password?",
  "body": "Hi, I forgot my password...",
  "sender_email": "user@example.com",
  "sender_tier": "free",
  "created_at": "2025-01-01T12:00:00",
  "previous_messages": [],
  "queue_size": 23,
  "task_id": "easy",
  "step_number": 1,
  "done": false
}
```

---

## 🎮 Action Space

The agent picks **one action per step**:

| Action Type | Fields | Description |
|---|---|---|
| `categorize` | `category` | Label ticket: billing / technical / account / feature_request / spam |
| `prioritize` | `priority` | Set priority: low / medium / high / urgent |
| `respond` | `response_text` | Draft a reply to the customer |
| `escalate` | `escalate_to` | Route to: tier2 / billing_team / engineering |
| `close` | `close_reason` | Close as: resolved / spam / duplicate |

---

## 🏆 Tasks

### Task 1 — Easy
**Goal:** Correctly categorize and close simple, unambiguous tickets (password reset, spam, basic billing).  
**Max steps:** 5  
**Baseline score:** ~0.72

### Task 2 — Medium
**Goal:** Handle tickets requiring correct escalation and priority judgment (API errors, feature requests, enterprise customers).  
**Max steps:** 8  
**Baseline score:** ~0.61

### Task 3 — Hard
**Goal:** Manage high-stakes enterprise tickets (data breach reports, legal cancellation demands) requiring correct triage, escalation, AND quality written responses covering required topics.  
**Max steps:** 12  
**Baseline score:** ~0.48

---

## 💰 Reward Function

Rewards are **shaped** — every step gives signal:

| Action | Reward Signal |
|---|---|
| Correct category | +1.0 |
| Wrong category | 0.0 |
| Correct priority | +1.0, partial credit by distance (±0.35/level) |
| Correct escalation team | +1.0 |
| Wrong team (but escalated) | +0.4 |
| Unnecessary escalation | 0.0 (penalty) |
| Response covering required topics | proportional to topic coverage |
| Premature close | 0.0 |

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/customer-support-triage
cd customer-support-triage
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t ticket-triage .
docker run -p 7860:7860 ticket-triage
```

### API Usage

```bash
# Reset environment
curl -X POST "http://localhost:7860/reset?task_id=easy"

# Take a step
curl -X POST "http://localhost:7860/step?task_id=easy" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "categorize", "category": "account"}'

# Get current state
curl "http://localhost:7860/state?task_id=easy"

# Get final grade
curl "http://localhost:7860/grade?task_id=easy"
```

---

## 🤖 Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"

python inference.py
```

### Expected Baseline Scores (gpt-4o-mini)

| Task | Score |
|---|---|
| Easy | ~0.72 |
| Medium | ~0.61 |
| Hard | ~0.48 |
| **Average** | **~0.60** |

---

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| POST | `/reset?task_id=` | Reset episode |
| POST | `/step?task_id=` | Submit action |
| GET | `/state?task_id=` | Current state |
| GET | `/grade?task_id=` | Final episode score |
| GET | `/tasks` | List all tasks |

---

## 📋 OpenEnv Spec Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `step()` → `(observation, reward, done, info)`
- ✅ `reset()` → initial observation
- ✅ `state()` → full internal state
- ✅ `openenv.yaml` with metadata
- ✅ 3 tasks with difficulty progression
- ✅ Deterministic graders (0.0–1.0)
- ✅ Shaped reward function
- ✅ Baseline inference script (`inference.py`)
- ✅ Dockerfile included
