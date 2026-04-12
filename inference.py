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
# Use abspath to handle edge cases where __file__ resolves to '' on import
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
# The evaluator calls /reset then /step repeatedly in sequence.
# We must persist the env instance across requests — creating a new
# env on every call loses all simulation state.
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
        # If no env exists yet (evaluator skipped /reset), initialise a default one
        if _env is None:
            _env = RailCascadeEnv(task="medium")
            _env.reset()

        # Parse action payload from request body
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Build StepActions from the request body.
        # Evaluator may send: {"actions": [...]} or a bare list, or an empty body.
        if isinstance(body, dict) and "actions" in body:
            raw_actions = body["actions"]
        elif isinstance(body, list):
            raw_actions = body
        else:
            raw_actions = []

        # Parse each individual action safely
        parsed_actions = []
        for item in raw_actions:
            try:
                parsed_actions.append(SingleAction(**item))
            except Exception:
                # Skip malformed action entries rather than crashing
                continue

        step_actions = StepActions(actions=parsed_actions)

        # Execute the simulation step
        obs, reward, done, info = _env.step(step_actions)

        # Serialise reward (Pydantic model)
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
# NO uvicorn.run() here. The Dockerfile CMD starts the server:
#   uvicorn inference:app --host 0.0.0.0 --port ${PORT:-7860}
# Running uvicorn.run() here causes [Errno 98] double-bind with the evaluator.