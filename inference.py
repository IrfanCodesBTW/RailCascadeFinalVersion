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
sys.path.insert(0, os.path.dirname(__file__))
from rail_cascade_env import RailCascadeEnv

# -------------------- APP ----------------------------
app = FastAPI()

# -------------------- RESET --------------------------
@app.post("/reset")
@app.get("/reset")
async def reset_endpoint(request: Request):
    print("🔥 RESET HIT 🔥")

    try:
        try:
            body = await request.json()
        except:
            body = {}

        task = body.get("task", "medium")

        valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
        if task not in valid_tasks:
            task = "medium"

        env = RailCascadeEnv(task=task)
        obs = env.reset()

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------- STEP ---------------------------
@app.post("/step")
async def step_endpoint(request: Request):
    print("🔥 STEP HIT 🔥")

    try:
        env = RailCascadeEnv(task="medium")
        obs = env.reset()

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------- HEALTH -------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    print("🚀 PURE OPENENV SERVER RUNNING 🚀")
    uvicorn.run(app, host="0.0.0.0", port=7860)