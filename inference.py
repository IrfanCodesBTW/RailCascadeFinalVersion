import json
import os
import sys
import random
import numpy as np

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import uvicorn

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- IMPORT ENV ---------------------
sys.path.insert(0, os.path.dirname(__file__))
from rail_cascade_env import RailCascadeEnv

# -------------------- APP ----------------------------
app = FastAPI(title="RailCascade OpenEnv Server")

# -------------------- OPENENV RESET ------------------
@app.post("/reset")
@app.get("/reset")
async def reset_endpoint(request: Request):
    print("🔥 OPENENV RESET HIT 🔥")

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

# -------------------- OPENENV STEP -------------------
@app.post("/step")
async def step_endpoint(request: Request):
    print("🔥 OPENENV STEP HIT 🔥")

    try:
        # minimal safe step
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

# -------------------- UI ROUTES (SAFE) ---------------
@app.post("/api/reset")
async def ui_reset():
    env = RailCascadeEnv(task="medium")
    obs = env.reset()

    return {
        "state": json.loads(json.dumps(obs.model_dump(), default=str))
    }

@app.post("/api/auto_step")
async def ui_step():
    return {"status": "ok"}

# -------------------- STATIC FILES -------------------
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- HEALTH -------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    print("🚀 OPENENV SERVER RUNNING 🚀")
    uvicorn.run(app, host="0.0.0.0", port=7860)