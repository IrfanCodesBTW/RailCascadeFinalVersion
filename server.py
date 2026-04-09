"""
RailCascade Mini V2 -- FastAPI Backend Server
Serves the environment state and wraps step/reset API for the frontend.
Session-based isolation: each /api/reset creates a unique session_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rail_cascade_env import (
    ALL_EDGES,
    ALL_NODES,
    NODE_POSITIONS,
    NODE_TYPES,
    TASK_CONFIGS,
    RailCascadeEnv,
    SingleAction,
    StepActions,
    greedy_agent,
)

# ---------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------

app = FastAPI(
    title="RailCascade Mini V2",
    description="Interactive railway traffic management RL environment",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Static files
# ---------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------
# Session-based environment storage
# ---------------------------------------------------------------

sessions: dict[str, RailCascadeEnv] = {}


# ---------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    session_id: str
    actions: list[dict[str, Any]] = []


class AutoStepRequest(BaseModel):
    session_id: str


class EnvResponse(BaseModel):
    state: dict
    reward: Union[float, dict]   # float for reset/state; float for step (breakdown in info)
    done: bool
    info: dict
    session_id: str = ""
    score: float = 0.0


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main frontend page."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/ping")
async def ping():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/api/graph")
async def get_graph():
    """Return the static graph topology with node positions."""
    return {
        "nodes": [
            {
                "id": node,
                "type": NODE_TYPES[node],
                "x": NODE_POSITIONS[node][0],
                "y": NODE_POSITIONS[node][1],
            }
            for node in ALL_NODES
        ],
        "edges": [
            {"from": src, "to": dst}
            for src, dst in ALL_EDGES
        ],
    }


@app.get("/api/tasks")
async def get_tasks():
    """Return available task configurations."""
    return {
        name: {
            "n_trains": cfg["n_trains"],
            "blocked_edges": [list(e) for e in cfg["blocked_edges"]],
            "max_steps": cfg["max_steps"],
        }
        for name, cfg in TASK_CONFIGS.items()
    }


@app.post("/reset")
@app.post("/api/reset")
@app.get("/reset")
@app.get("/api/reset")
async def reset_env(request: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment with the specified task. Returns a session_id."""
    if request.task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task}'. Choose from: {list(TASK_CONFIGS.keys())}",
        )

    sid = str(uuid4())
    sessions[sid] = RailCascadeEnv(task=request.task)
    sessions[sid].reset()

    _score = sessions[sid].get_score()
    return EnvResponse(
        state=sessions[sid].state(),
        reward=0.0,
        done=sessions[sid].done,
        info={},
        session_id=sid,
        score=_score,
    )


@app.get("/state")
@app.get("/api/state")
async def get_state(session_id: str):
    """Return the current environment state for a session."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /api/reset first.")

    return EnvResponse(
        state=env.state(),
        reward=0.0,
        done=env.done,
        info={},
        session_id=session_id,
        score=env.get_score(),
    )


@app.post("/step")
@app.post("/api/step")
async def step_env(request: StepRequest):
    """Apply actions and advance one timestep."""
    env = sessions.get(request.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /api/reset first.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /api/reset to start a new episode.")

    # Parse actions
    parsed_actions = []
    for act_dict in request.actions:
        try:
            parsed_actions.append(SingleAction(**act_dict))
        except Exception as e:
            # Skip invalid actions
            pass

    step_actions = StepActions(actions=parsed_actions)
    obs, reward, done, info = env.step(step_actions)

    return EnvResponse(
        state=env.state(),
        reward=reward.step_reward,
        done=done,
        info={**info, "reward_breakdown": reward.reward_breakdown, "total_delay": reward.total_delay},
        session_id=request.session_id,
        score=env.get_score(),
    )


@app.post("/api/auto_step")
async def auto_step(request: AutoStepRequest):
    """Auto-step using the greedy baseline agent."""
    env = sessions.get(request.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /api/reset first.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /api/reset to start a new episode.")

    agent_actions = greedy_agent(env)
    obs, reward, done, info = env.step(agent_actions)

    return EnvResponse(
        state=env.state(),
        reward=reward.step_reward,
        done=done,
        info={**info, "reward_breakdown": reward.reward_breakdown, "total_delay": reward.total_delay},
        session_id=request.session_id,
        score=env.get_score(),
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session."""
    sessions.pop(session_id, None)
    return {"deleted": session_id}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("RailCascade Mini V2 -- Server")
    print("Open http://localhost:7860 in your browser")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
