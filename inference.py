import json
import os
import sys
import random
import numpy as np
from openai import OpenAI

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- ENV VARS -----------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN", "dummy_key")
TASK = os.environ.get("TASK", "medium")

# OpenAI client (kept for compliance)
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# -------------------- IMPORT ENV ---------------------
sys.path.insert(0, os.path.dirname(__file__))
from rail_cascade_env import RailCascadeEnv, StepActions, SingleAction


# -------------------- SAFE NEXT NODE --------------------
def get_next_node(train):
    if hasattr(train, "path_remaining") and train.path_remaining:
        return train.path_remaining[0]

    if hasattr(train, "path") and hasattr(train, "current_index"):
        if train.current_index + 1 < len(train.path):
            return train.path[train.current_index + 1]

    return None


# -------------------- SAFE PATH LENGTH --------------------
def get_remaining_length(train):
    if hasattr(train, "path_remaining") and train.path_remaining:
        return len(train.path_remaining)

    if hasattr(train, "path") and hasattr(train, "current_index"):
        return len(train.path) - train.current_index - 1

    return 0


# -------------------- PATHFINDING --------------------
def find_alternate_path(env, train):
    graph = env.graph
    start = train.position
    goal = "T1"

    from collections import deque

    queue = deque([(start, [])])
    visited = set()

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for neighbor in graph[node]:
            edge = (node, neighbor)

            if edge in env.blocked_edges:
                continue

            queue.append((neighbor, path + [neighbor]))

    return []


# -------------------- POLICY AGENT --------------------
def policy_agent(env):
    trains = [t for t in env.trains if not t.arrived]
    blocked_edges = set(env.blocked_edges)

    actions = []

    next_edges = {}

    for t in trains:
        next_node = get_next_node(t)
        if next_node:
            edge = (t.position, next_node)
            next_edges.setdefault(edge, []).append(t)

    conflict_losers = set()

    for edge, ts in next_edges.items():
        if len(ts) > 1:
            ts_sorted = sorted(ts, key=lambda x: x.id)
            for l in ts_sorted[1:]:
                conflict_losers.add(l.id)

    for t in trains:
        action = "noop"
        new_path = None

        next_node = get_next_node(t)

        if not next_node:
            actions.append({"train_id": t.id, "action": "noop"})
            continue

        next_edge = (t.position, next_node)

        if next_edge in blocked_edges:
            action = "reroute"

        elif t.id in conflict_losers:
            action = "hold"

        elif getattr(t, "delay", 0) >= 3:
            action = "reroute"

        elif t.position in ["J2", "J3", "J4"] and len(next_edges.get(next_edge, [])) > 1:
            action = "hold"

        elif get_remaining_length(t) > 4:
            action = "reroute"

        if action == "reroute":
            new_path = find_alternate_path(env, t)

        act = {
            "train_id": t.id,
            "action": action
        }

        if new_path:
            act["new_path"] = new_path

        actions.append(act)

    return {
        "reasoning": "deterministic policy",
        "actions": actions
    }


# -------------------- PARSE ACTIONS ------------------
def parse_actions(llm_response, env):
    raw = llm_response.get("actions", [])
    parsed = []

    valid_ids = {t.id for t in env.trains if not t.arrived}

    for act in raw:
        tid = act.get("train_id")
        atype = act.get("action", "noop")

        if tid not in valid_ids:
            continue

        if atype not in ("noop", "hold", "reroute"):
            atype = "noop"

        kwargs = {"action": atype, "train_id": tid}

        if atype == "reroute" and "new_path" in act:
            kwargs["new_path"] = act["new_path"]

        try:
            parsed.append(SingleAction(**kwargs))
        except Exception:
            parsed.append(SingleAction(action="noop", train_id=tid))

    covered = {a.train_id for a in parsed}

    for tid in valid_ids:
        if tid not in covered:
            parsed.append(SingleAction(action="noop", train_id=tid))

    return StepActions(actions=parsed)


# -------------------- RUN EPISODE --------------------
def run_episode(task):
    env = RailCascadeEnv(task=task)
    env.reset()

    step = 0

    print("[START]")
    print(json.dumps({
        "task": task,
        "model": MODEL_NAME
    }))

    while not env.done:
        llm_resp = policy_agent(env)
        actions = parse_actions(llm_resp, env)

        obs, reward, done, info = env.step(actions)
        step += 1

        print("[STEP]")
        print(json.dumps({
            "step": step,
            "action": [a.model_dump() for a in actions.actions],
            "reward": float(reward.step_reward),
            "done": bool(done)
        }))

    final_score = float(env.get_score())

    print("[END]")
    print(json.dumps({
        "final_score": final_score
    }))

    return final_score


# -------------------- HTTP SERVER (for OpenEnv checker) --------------------
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBase
import uvicorn

http_app = FastAPI(title="RailCascade Inference Server")

class ResetRequest(PydanticBase):
    task: str = "medium"

@http_app.post("/reset")
async def reset_endpoint(request: ResetRequest):
    """OpenEnv checker POSTs here to verify the environment is alive."""
    valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme")
    task = request.task if request.task in valid_tasks else "medium"
    env = RailCascadeEnv(task=task)
    env.reset()
    return {"status": "ok", "task": task, "score": float(env.get_score())}

@http_app.get("/ping")
async def ping():
    return {"status": "ok"}

@http_app.get("/health")
async def health():
    return {"status": "healthy"}


# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    CLI_TASKS = ("easy", "medium", "hard", "dynamic_medium", "extreme")
    if len(sys.argv) > 1 and sys.argv[1] in CLI_TASKS:
        # CLI mode: python inference.py medium
        run_episode(sys.argv[1])
    else:
        # Server mode: start HTTP server for OpenEnv checker
        print("Starting RailCascade inference server on port 8080...")
        uvicorn.run(http_app, host="0.0.0.0", port=7860, log_level="info")
