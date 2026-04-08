"""
server/app.py — Required entry point for OpenEnv multi-mode deployment.
This re-exports the main FastAPI app.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import TicketTriageEnv, Action, TASK_CONFIGS
from datetime import datetime

app = FastAPI(
    title="Customer Support Ticket Triage — OpenEnv",
    description="An OpenEnv environment where agents learn to triage customer support tickets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, TicketTriageEnv] = {
    "easy": TicketTriageEnv("easy"),
    "medium": TicketTriageEnv("medium"),
    "hard": TicketTriageEnv("hard"),
}


def _get_env(task_id: str) -> TicketTriageEnv:
    if task_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'. Use: easy, medium, hard")
    return _envs[task_id]


@app.get("/")
def root():
    return {
        "name": "customer-support-triage",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "spec": "openenv-v1",
        "description": "AI agent environment for customer support ticket triage"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = "easy"):
    env = _get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: Action, task_id: str = "easy"):
    env = _get_env(task_id)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "easy"):
    env = _get_env(task_id)
    return env.state()


@app.get("/grade")
def grade(task_id: str = "easy"):
    env = _get_env(task_id)
    score = env.grade()
    return {"task_id": task_id, "score": score}


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "num_tickets": len(cfg["tickets"]),
        }
        for tid, cfg in TASK_CONFIGS.items()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
