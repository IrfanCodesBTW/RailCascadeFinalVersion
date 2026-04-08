import json
import os
import sys
import time
import random
import numpy as np
from openai import OpenAI

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- ENV VARS -----------------------
API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://api.openai.com/v1"
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN")  # OpenAI key
TASK = os.environ.get("TASK", "medium")
MAX_RETRIES = 3

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# -------------------- IMPORT ENV ---------------------
sys.path.insert(0, os.path.dirname(__file__))
from rail_cascade_env import RailCascadeEnv, StepActions, SingleAction


SYSTEM_PROMPT = """You are a railway traffic controller AI.
Return ONLY JSON with actions."""


def build_prompt(state, timestep, max_steps):
    return f"""
Timestep: {timestep}/{max_steps}
State:
{json.dumps(state)}

Return JSON:
{{"reasoning": "...", "actions": [{{"action": "noop", "train_id": 0}}]}}
"""


def call_llm(prompt, env):
    for _ in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            text = resp.choices[0].message.content.strip()

            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    inner = parts[1]
                    if inner.startswith("json"):
                        inner = inner[4:]
                    text = inner.strip()

            return json.loads(text)

        except Exception:
            time.sleep(0.5)

    return {
        "reasoning": "fallback",
        "actions": [
            {"action": "noop", "train_id": t.id}
            for t in env.trains if not t.arrived
        ]
    }


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

        kwargs = {"action_type": atype, "train_id": tid}

        if atype == "reroute" and "new_path" in act:
            kwargs["new_path"] = act["new_path"]

        try:
            parsed.append(SingleAction(**kwargs))
        except Exception:
            parsed.append(SingleAction(action_type="noop", train_id=tid))

    covered = {a.train_id for a in parsed}
    for tid in valid_ids:
        if tid not in covered:
            parsed.append(SingleAction(action_type="noop", train_id=tid))

    return StepActions(actions=parsed)


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
        state = env.state()

        prompt = build_prompt(state, env.timestep, env.max_steps)
        llm_resp = call_llm(prompt, env)
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


if __name__ == "__main__":
    if not API_KEY:
        print("[ERROR] Missing OpenAI API key (HF_TOKEN).")
        sys.exit(1)

    task = sys.argv[1] if len(sys.argv) > 1 else TASK
    run_episode(task)