import json
import os
import sys
import random
import numpy as np
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBase
import uvicorn

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
def build_prompt(env) -> str:
    """Build a concise state prompt for the LLM."""
    trains = [t for t in env.trains if not t.arrived]
    train_lines = []
    for t in trains:
        next_node = get_next_node(t)
        train_lines.append(
            f"  Train {t.id}: pos={t.position}, delay={getattr(t,'delay',0)}, "
            f"status={getattr(t,'status','moving')}, next={next_node}"
        )
    blocked = list(env.blocked_edges)
    return f"""You are controlling a railway traffic management system.

Graph: 10-node directed graph. Terminal is T1. Sources are S1, S2.
Junctions: J1, J2, J3, J4. Corridors: C1, C2, C3.

Blocked edges: {blocked}
Time step: {env.timestep}

Active trains:
{chr(10).join(train_lines)}

For each active train, output one action:
- "noop": continue on current path
- "hold": wait one step (use when conflict ahead)
- "reroute": take alternate path (must include new_path as list of node IDs to T1)

Respond ONLY with a valid JSON object in this exact format:
{{
  "reasoning": "brief explanation",
  "actions": [
    {{"train_id": 0, "action": "noop"}},
    {{"train_id": 1, "action": "hold"}},
    {{"train_id": 2, "action": "reroute", "new_path": ["J4", "J3", "T1"]}}
  ]
}}

Rules:
- Every active train must have exactly one action
- Only use reroute if you include a valid new_path that leads to T1
- Prioritize trains with high delay for rerouting
- Hold trains that would collide on the same edge
"""


def deterministic_fallback(env) -> dict:
    """Original rule-based policy used as fallback if LLM fails."""
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
        if action == "reroute":
            new_path = find_alternate_path(env, t)
        act = {"train_id": t.id, "action": action}
        if new_path:
            act["new_path"] = new_path
        actions.append(act)
    return {"reasoning": "deterministic fallback", "actions": actions}


def llm_agent(env) -> dict:
    """Call the LLM API for decisions. Falls back to deterministic policy on error."""
    try:
        prompt = build_prompt(env)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        # Validate it has actions key
        if "actions" not in result:
            raise ValueError("No actions key in LLM response")
        return result
    except Exception as e:
        print(f"[LLM fallback] {e}", file=sys.stderr)
        return deterministic_fallback(env)


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
        llm_resp = llm_agent(env)
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
http_app = FastAPI(title="RailCascade Inference Server")

class ResetRequest(PydanticBase):
    task: str = "medium"

@http_app.post("/reset")
async def reset_endpoint(request: ResetRequest):
    valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
    task = request.task if request.task in valid_tasks else "medium"
    try:
        env = RailCascadeEnv(task=task)
        obs = env.reset()
        return {
            "status": "ok",
            "task": task,
            "score": float(env.get_score()),
            "n_trains": len(env.trains),
            "blocked_edges": [list(e) for e in env.blocked_edges],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}, 500

@http_app.get("/ping")
async def ping():
    return {"status": "ok"}

@http_app.get("/health")
async def health():
    return {"status": "healthy"}


# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    CLI_TASKS = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
    if len(sys.argv) > 1 and sys.argv[1] in CLI_TASKS:
        run_episode(sys.argv[1])
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Running benchmark across all tasks...")
        results = {}
        for task in CLI_TASKS:
            print(f"\n--- Task: {task} ---")
            score = run_episode(task)
            results[task] = round(score, 4)
        print("\n[BENCHMARK SUMMARY]")
        print(json.dumps(results, indent=2))
    else:
        print("Starting RailCascade inference server on port 7860...")
        uvicorn.run(http_app, host="0.0.0.0", port=7860, log_level="info")
