"""
Customer Support Ticket Triage Environment
OpenEnv-compliant environment for AI agent training and evaluation.
"""

import random
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


def clamp(v: float) -> float:
    """Ensure value is strictly between 0 and 1."""
    return round(max(0.01, min(0.99, float(v))), 4)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    sender_email: str
    sender_tier: str
    created_at: str
    previous_messages: List[Dict[str, str]] = Field(default_factory=list)
    queue_size: int
    task_id: str
    step_number: int
    done: bool = False

class Action(BaseModel):
    action_type: str
    category: Optional[str] = None
    priority: Optional[str] = None
    response_text: Optional[str] = None
    escalate_to: Optional[str] = None
    close_reason: Optional[str] = None

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    message: str


# ─── Ticket Bank ──────────────────────────────────────────────────────────────

TICKETS = {
    "easy": [
        {
            "ticket_id": "TKT-1001",
            "subject": "How do I reset my password?",
            "body": "Hi, I forgot my password and can't log in. Can you help me reset it? Thanks.",
            "sender_email": "user@example.com",
            "sender_tier": "free",
            "ground_truth": {
                "category": "account",
                "priority": "medium",
                "requires_escalation": False,
                "should_close": True,
                "key_topics": ["password", "reset", "login"]
            }
        },
        {
            "ticket_id": "TKT-1002",
            "subject": "FREE MONEY CLICK HERE!!!",
            "body": "Congratulations! You have won $1,000,000. Click this link: http://spam.example.com",
            "sender_email": "spammer@suspicious.xyz",
            "sender_tier": "free",
            "ground_truth": {
                "category": "spam",
                "priority": "low",
                "requires_escalation": False,
                "should_close": True,
                "key_topics": ["spam"]
            }
        },
        {
            "ticket_id": "TKT-1003",
            "subject": "Invoice question",
            "body": "I was charged twice for my subscription this month. Please refund the duplicate charge.",
            "sender_email": "customer@business.com",
            "sender_tier": "pro",
            "ground_truth": {
                "category": "billing",
                "priority": "high",
                "requires_escalation": True,
                "escalate_to": "billing_team",
                "should_close": False,
                "key_topics": ["charged", "twice", "refund", "duplicate"]
            }
        }
    ],
    "medium": [
        {
            "ticket_id": "TKT-2001",
            "subject": "API returning 500 errors intermittently",
            "body": (
                "Hello, since yesterday our integration has been getting intermittent 500 errors "
                "from your API. This is affecting our production system. Error code: ERR_INTERNAL_5XX. "
                "This is critical for our business operations. We are an enterprise customer."
            ),
            "sender_email": "devops@bigcorp.com",
            "sender_tier": "enterprise",
            "ground_truth": {
                "category": "technical",
                "priority": "urgent",
                "requires_escalation": True,
                "escalate_to": "engineering",
                "should_close": False,
                "key_topics": ["500 errors", "API", "production", "intermittent", "enterprise"]
            }
        },
        {
            "ticket_id": "TKT-2002",
            "subject": "Feature request: dark mode",
            "body": (
                "I'd love to see a dark mode option in the dashboard. Many of my colleagues "
                "have also requested this. It would make working late at night much easier. "
                "Is this on the roadmap?"
            ),
            "sender_email": "user2@startup.io",
            "sender_tier": "pro",
            "ground_truth": {
                "category": "feature_request",
                "priority": "low",
                "requires_escalation": False,
                "should_close": True,
                "key_topics": ["dark mode", "feature", "roadmap", "dashboard"]
            }
        }
    ],
    "hard": [
        {
            "ticket_id": "TKT-3001",
            "subject": "Data breach concern - urgent",
            "body": (
                "I noticed that when I log into my account, I can see another user's data. "
                "I saw their name, email, and billing information. This is a serious security issue. "
                "I am an enterprise customer and this is unacceptable. My account ID is ENT-44821. "
                "I expect immediate response and a full incident report."
            ),
            "sender_email": "cto@enterprise-client.com",
            "sender_tier": "enterprise",
            "ground_truth": {
                "category": "technical",
                "priority": "urgent",
                "requires_escalation": True,
                "escalate_to": "engineering",
                "should_close": False,
                "requires_response": True,
                "response_must_include": ["security", "urgency", "investigate", "apologize"],
                "key_topics": ["data breach", "security", "other user data", "enterprise", "urgent"]
            }
        },
        {
            "ticket_id": "TKT-3002",
            "subject": "Cancellation and refund request",
            "body": (
                "I want to cancel my annual enterprise subscription immediately. "
                "We signed up 2 months ago and were promised certain features that have not been delivered. "
                "Per our contract clause 7.3, we are entitled to a pro-rated refund. "
                "I need a response within 24 hours or I will escalate to our legal team."
            ),
            "sender_email": "procurement@large-enterprise.com",
            "sender_tier": "enterprise",
            "ground_truth": {
                "category": "billing",
                "priority": "urgent",
                "requires_escalation": True,
                "escalate_to": "billing_team",
                "should_close": False,
                "requires_response": True,
                "response_must_include": ["refund", "contract", "understand", "escalate"],
                "key_topics": ["cancel", "refund", "enterprise", "contract", "legal", "annual"]
            }
        }
    ]
}

TASK_CONFIGS = {
    "easy": {
        "description": "Categorize and handle simple, unambiguous support tickets",
        "max_steps": 5,
        "tickets": TICKETS["easy"],
        "scoring_weights": {"category": 0.4, "priority": 0.3, "action": 0.3}
    },
    "medium": {
        "description": "Handle tickets requiring correct escalation and priority judgment",
        "max_steps": 8,
        "tickets": TICKETS["medium"],
        "scoring_weights": {"category": 0.3, "priority": 0.3, "escalation": 0.4}
    },
    "hard": {
        "description": "Manage high-stakes enterprise tickets requiring correct triage, escalation, and quality responses",
        "max_steps": 12,
        "tickets": TICKETS["hard"],
        "scoring_weights": {"category": 0.2, "priority": 0.2, "escalation": 0.25, "response": 0.35}
    }
}


# ─── Environment ──────────────────────────────────────────────────────────────

class TicketTriageEnv:
    def __init__(self, task_id: str = "easy"):
        assert task_id in TASK_CONFIGS
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]
        self._state: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> Observation:
        ticket = random.choice(self.config["tickets"])
        self._state = {
            "task_id": self.task_id,
            "ticket": ticket,
            "step_number": 0,
            "done": False,
            "actions_taken": [],
            "category_set": None,
            "priority_set": None,
            "escalated": False,
            "escalate_to": None,
            "responded": False,
            "response_text": "",
            "closed": False,
            "close_reason": None,
            "queue_size": random.randint(10, 50),
            "cumulative_reward": 0.5,  # Start at neutral, not 0.0
        }
        return self._make_observation()

    def step(self, action: Action):
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._state["step_number"] += 1
        self._state["actions_taken"].append(action.model_dump())

        reward = self._compute_reward(action)
        self._apply_action(action)

        done = self._check_done()
        self._state["done"] = done

        obs = self._make_observation()
        info = {
            "step": self._state["step_number"],
            "actions_taken": len(self._state["actions_taken"]),
            "cumulative_reward": clamp(self._state["cumulative_reward"]),
        }
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    def _make_observation(self) -> Observation:
        s = self._state
        t = s["ticket"]
        return Observation(
            ticket_id=t["ticket_id"],
            subject=t["subject"],
            body=t["body"],
            sender_email=t["sender_email"],
            sender_tier=t["sender_tier"],
            created_at=datetime.utcnow().isoformat(),
            previous_messages=[],
            queue_size=s["queue_size"],
            task_id=s["task_id"],
            step_number=s["step_number"],
            done=s["done"],
        )

    def _apply_action(self, action: Action):
        s = self._state
        if action.action_type == "categorize" and action.category:
            s["category_set"] = action.category
        elif action.action_type == "prioritize" and action.priority:
            s["priority_set"] = action.priority
        elif action.action_type == "respond" and action.response_text:
            s["responded"] = True
            s["response_text"] = action.response_text
        elif action.action_type == "escalate":
            s["escalated"] = True
            s["escalate_to"] = action.escalate_to
        elif action.action_type == "close":
            s["closed"] = True
            s["close_reason"] = action.close_reason

    def _check_done(self) -> bool:
        s = self._state
        if s["step_number"] >= self.config["max_steps"]:
            return True
        gt = s["ticket"]["ground_truth"]
        if s["closed"] or (s["escalated"] and not gt.get("requires_response", False)):
            return True
        if s["escalated"] and s["responded"]:
            return True
        return False

    def _compute_reward(self, action: Action) -> Reward:
        s = self._state
        gt = s["ticket"]["ground_truth"]
        weights = self.config["scoring_weights"]
        breakdown = {}
        messages = []

        if action.action_type == "categorize":
            correct = action.category == gt["category"]
            breakdown["category"] = 0.92 if correct else 0.08
            messages.append(f"Category {'correct' if correct else 'wrong'}")

        elif action.action_type == "prioritize":
            pm = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
            got = pm.get(action.priority, -1)
            exp = pm.get(gt["priority"], -1)
            dist = abs(got - exp)
            score = clamp(0.92 - dist * 0.28)
            breakdown["priority"] = score
            messages.append(f"Priority score {score:.2f}")

        elif action.action_type == "escalate":
            should_escalate = gt.get("requires_escalation", False)
            correct_team = gt.get("escalate_to", None)
            if should_escalate:
                team_score = 0.92 if action.escalate_to == correct_team else 0.45
                breakdown["escalation"] = clamp(team_score)
                messages.append(f"Escalation correct, team score {team_score:.2f}")
            else:
                breakdown["escalation"] = 0.08
                messages.append("Unnecessary escalation")

        elif action.action_type == "respond":
            text = (action.response_text or "").lower()
            required_topics = gt.get("response_must_include", [])
            if required_topics:
                hits = sum(1 for kw in required_topics if kw in text)
                score = clamp(0.08 + (hits / len(required_topics)) * 0.84)
            else:
                score = 0.65 if len(text) > 50 else 0.35
            breakdown["response"] = clamp(score)
            messages.append(f"Response quality score {score:.2f}")

        elif action.action_type == "close":
            should_close = gt.get("should_close", False)
            if should_close:
                correct_reason = action.close_reason == ("spam" if gt["category"] == "spam" else "resolved")
                breakdown["close"] = 0.92 if correct_reason else 0.55
                messages.append(f"Close {'correct' if correct_reason else 'suboptimal'}")
            else:
                breakdown["close"] = 0.08
                messages.append("Premature close")

        else:
            breakdown["unknown"] = 0.08
            messages.append(f"Unknown action: {action.action_type}")

        total = sum(breakdown[k] * weights.get(k, 0.1) for k in breakdown)
        total = clamp(total)
        s["cumulative_reward"] = clamp(s["cumulative_reward"] * 0.9 + total * 0.1)

        return Reward(
            value=total,
            breakdown={k: clamp(v) for k, v in breakdown.items()},
            message=" | ".join(messages)
        )

    def grade(self) -> float:
        s = self._state
        gt = s["ticket"]["ground_truth"]
        scores = []

        # Category score
        if s["category_set"] is not None:
            scores.append(0.92 if s["category_set"] == gt["category"] else 0.08)
        else:
            scores.append(0.08)

        # Priority score
        if s["priority_set"] is not None:
            pm = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
            dist = abs(pm.get(s["priority_set"], -1) - pm.get(gt["priority"], -1))
            scores.append(clamp(0.92 - dist * 0.28))
        else:
            scores.append(0.08)

        # Escalation score
        should_esc = gt.get("requires_escalation", False)
        if should_esc:
            if s["escalated"]:
                scores.append(0.92 if s["escalate_to"] == gt.get("escalate_to") else 0.45)
            else:
                scores.append(0.08)
        else:
            scores.append(0.08 if s["escalated"] else 0.92)

        # Response score (hard task)
        if self.task_id == "hard":
            required = gt.get("response_must_include", [])
            if required and s["responded"]:
                text = s["response_text"].lower()
                hits = sum(1 for kw in required if kw in text)
                scores.append(clamp(0.08 + (hits / len(required)) * 0.84))
            elif required and not s["responded"]:
                scores.append(0.08)
            else:
                scores.append(0.82)

        final = sum(scores) / len(scores)
        return clamp(final)
