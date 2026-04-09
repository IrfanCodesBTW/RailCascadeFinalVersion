import json
import os
import sys
import random
import numpy as np
from openai import OpenAI
from fastapi import FastAPI, Body
import uvicorn

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- ENV VARS -----------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN")

# -------------------- OPENAI (SAFE INIT) -------------
client = None
if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"[OpenAI Init Failed] {e}")
        client = None

# -------------------- IMPORT ENV ---------------------
sys.path.insert(0, os.path.dirname(__file__))
from rail_cascade_env import RailCascadeEnv

# -------------------- FASTAPI ------------------------
http_app = FastAPI(title="RailCascade Inference Server")

# -------------------- RESET --------------------------
@http_app.post("/reset")
@http_app.get("/reset")
async def reset_endpoint(request: dict | None = Body(default=None)):
    request = request or {}

    task = request.get("task", "medium")

    valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
    if task not in valid_tasks:
        task = "medium"

    try:
        env = RailCascadeEnv(task=task)
        obs = env.reset()

        return {
            "status": "ok",
            "task": task,
            "score": float(env.get_score()),
            "n_trains": len(env.trains),
            "blocked_edges": [list(e) for e in env.blocked_edges],
            "state": json.loads(json.dumps(obs.model_dump(), default=str))
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}

# -------------------- STEP (MINIMAL SAFE) ------------
@http_app.post("/step")
async def step_endpoint(request: dict = Body(...)):
    try:
        # Minimal valid step (no crashes)
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
@http_app.get("/ping")
async def ping():
    return {"status": "ok"}

@http_app.get("/health")
async def health():
    return {"status": "healthy"}

# -------------------- MAIN ---------------------------
if __name__ == "__main__":
    print("Starting RailCascade inference server on port 7860...")
    uvicorn.run(http_app, host="0.0.0.0", port=7860, log_level="info")