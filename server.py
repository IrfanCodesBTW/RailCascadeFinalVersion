from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# ---- STATE ----
state = {
    "trains": [],
    "time": 0
}

# ---- MODELS ----
class StepRequest(BaseModel):
    action: dict

# ---- ROUTES ----

@app.get("/")
def root():
    return {"message": "OpenEnv Railway Env Running"}

# ✅ FIXED: NO required body
@app.post("/reset")
def reset():
    global state
    state = {
        "trains": [],
        "time": 0
    }

    return {
        "state": state,
        "info": {}
    }

# Step requires action (correct behavior)
@app.post("/step")
def step(req: StepRequest):
    global state

    action = req.action

    # Example logic (replace with your actual logic)
    state["time"] += 1

    return {
        "state": state,
        "reward": 0,
        "done": False,
        "info": {}
    }

# Optional but useful
@app.get("/state")
def get_state():
    return state


# ---- RUN ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)