import json
import os
import sys
import random
import numpy as np

from fastapi import FastAPI, Request
import uvicorn

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- IMPORT ENV ---------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rail_cascade_env import RailCascadeEnv, StepActions, SingleAction

# -------------------- APP ----------------------------
app = FastAPI()

# -------------------- ROOT ---------------------------
@app.get("/")
async def root():
    return {"message": "RailCascade API is running", "status": "online"}

# -------------------- GLOBAL STATE ------------------
_env: RailCascadeEnv | None = None

# -------------------- RESET --------------------------
@app.post("/reset")
@app.get("/reset")
async def reset_endpoint(request: Request):
    print("🔥 RESET HIT 🔥")
    global _env

    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        task = body.get("task", "medium") if isinstance(body, dict) else "medium"

        valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
        if task not in valid_tasks:
            task = "medium"

        _env = RailCascadeEnv(task=task)
        obs = _env.reset()

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "state": None,
            "reward": 0.0,
            "done": True,
            "info": {}
        }

# -------------------- STEP ---------------------------
@app.post("/step")
async def step_endpoint(request: Request):
    print("🔥 STEP HIT 🔥")
    global _env

    try:
        if _env is None:
            _env = RailCascadeEnv(task="medium")
            _env.reset()

        try:
            body = await request.json()
        except Exception:
            body = {}

        if isinstance(body, dict) and "actions" in body:
            raw_actions = body["actions"]
        elif isinstance(body, list):
            raw_actions = body
        else:
            raw_actions = []

        parsed_actions = []
        for item in raw_actions:
            try:
                parsed_actions.append(SingleAction(**item))
            except Exception:
                continue

        step_actions = StepActions(actions=parsed_actions)
        obs, reward, done, info = _env.step(step_actions)

        try:
            reward_dict = reward.model_dump()
        except Exception:
            reward_dict = {"step_reward": 0.0, "reward_breakdown": {}, "total_delay": 0}

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": json.loads(json.dumps(reward_dict, default=str)),
            "done": bool(done),
            "info": info if isinstance(info, dict) else {}
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "state": None,
            "reward": 0.0,
            "done": True,
            "info": {}
        }

# -------------------- HEALTH -------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# -------------------- MAIN ---------------------------
# PORT is injected by the OpenEnv/HuggingFace evaluator at runtime.
# We NEVER hardcode 7860 — we read the env var and fall back to 7860
# only for local development. The __name__ guard ensures uvicorn is
# NEVER started when openenv-core imports this module directly.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 RailCascade SERVER STARTING ON PORT {port} 🚀")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )