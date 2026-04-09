---
title: RailCascade Mini V2
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---

# RailCascade Mini V2

**Deterministic Railway Traffic Management RL Environment**

A 10-node directed graph network where an AI agent must coordinate multiple trains, issuing per-train hold/reroute/noop commands each timestep to minimize total delay caused by blocked edges, edge conflicts, and node-occupancy cascading.

---

## Architecture

```
RailCascadeV2/
  rail_cascade_env.py    # Core environment (reset/step/state/get_score)
  server.py              # FastAPI backend serving REST API + frontend
  static/
    index.html           # Dashboard page
    style.css            # Dark glassmorphism theme
    app.js               # Canvas rendering + API integration
  inference.py           # LLM-based agent (Gemini via OpenAI-compatible API)
  openenv.yaml           # OpenEnv configuration
  requirements.txt       # Python dependencies
  Dockerfile             # Container build
  README.md              # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run sanity checks (9 tests across all tasks)
python rail_cascade_env.py

# Run example episodes with greedy baseline
python rail_cascade_env.py run all

# Start the interactive dashboard
python server.py
# Open http://localhost:7860
```

## Running the LLM Agent

```bash
# Set your API key (Gemini free tier works)
# Get a free Gemini key at: https://aistudio.google.com
export HF_TOKEN="your-gemini-api-key-here"

# Run on any task
python inference.py easy
python inference.py medium
python inference.py hard
python inference.py dynamic_medium
python inference.py extreme

# Run full benchmark across all tasks
python inference.py benchmark
```

Logs are structured JSONL:
- `[START]` -- episode initialized
- `[STEP]`  -- per-timestep action + reward
- `[END]`   -- final score + arrival stats

## Graph Topology

10 nodes (2 sources, 4 junctions, 3 corridors, 1 terminal) connected by 14 directed edges:

```
         S2
          |
          v
S1 -> J1 -> J2 -> J3 -> T1
        \    |  \    |  /
         C1  C2  J4  C3
          \  |  / \  /
           > J2  J3 <
```

**Node Types:**
- **Source** (S1, S2): Train starting points
- **Junction** (J1-J4): Branching/merging points
- **Corridor** (C1-C3): Bypass routes
- **Terminal** (T1): Destination for all trains

## Environment API

### `reset() -> ObservationState`
Initialize the environment. Sets blocked edges and computes initial BFS paths for all trains.

### `step(StepActions) -> (ObservationState, StepReward, bool, dict)`
Execute one timestep with 6 transition phases:
1. **Apply Actions** - Hold/reroute/noop per train
2. **Collect Intents** - Determine which trains want to move (with stranded train recovery)
3. **Blocked Edge Check** - Trains on blocked edges get delay
4. **Edge Conflict Resolution** - Same edge, lowest ID wins
5. **Node Occupancy Cascade** - Blocked trains block trains behind them
6. **Execute Movement** - Remaining trains advance
7. **Dynamic Blocking** - Mid-episode edge blocks + automatic rerouting

### `state() -> dict`
Full serializable snapshot including graph, trains, and metadata.

### `get_score() -> float`
Deterministic final score in [0.0, 1.0]:
```
score = delay_score * arrival_ratio
delay_score = 1.0 - total_delay / sum_optimal_path_lengths
arrival_ratio = arrived_count / n_trains
```

## Action Space

Per-train actions submitted as a list:
```json
{
  "actions": [
    {"action": "noop", "train_id": 0},
    {"action": "hold", "train_id": 1},
    {"action": "reroute", "train_id": 2, "new_path": ["J4", "J3", "T1"]}
  ]
}
```

## Task Levels

| Task             | Trains | Blocked Edges           | Dynamic | Max Steps | Difficulty |
|------------------|--------|-------------------------|---------|-----------|------------|
| easy             | 3      | J2->J3                  | No      | 20        | 0.2        |
| medium           | 6      | J2->J3, J1->C1, J4->T1 | No      | 30        | 0.5        |
| hard             | 8      | J2->J3, J4->T1, J1->C1 | No      | 40        | 0.9        |
| dynamic_medium   | 8      | J2->J3 (+every 3 steps) | Yes     | 50        | 0.7        |
| extreme          | 10     | J2->J3, J4->T1 (+every 4 steps) | Yes | 60    | 1.0        |
| vip_routing      | 6      | J2->J3, J1->C1          | No      | 30        | 0.6        |

## Benchmark Results

| Task             | Noop  | Greedy | Gemini 2.0 Flash | Human |
|------------------|-------|--------|------------------|-------|
| easy             | ~0.80 | ~0.90  | 0.8000           | TBD   |
| medium           | ~0.46 | ~0.73  | 0.4615           | TBD   |
| hard             | ~0.00 | ~0.00  | 0.0000           | TBD   |
| dynamic_medium   | TBD   | TBD    | 0.0000           | TBD   |
| extreme          | TBD   | TBD    | 0.0000           | TBD   |
| vip_routing      | ~0.30 | ~0.30  | 0.3000           | TBD   |

Run `python inference.py benchmark` to fill in the TBD values with your results.

## Interactive Dashboard

The web frontend provides:
- **Canvas graph visualization** with type-colored nodes and curved directed edges
- **Real-time train animation** with smooth interpolation
- **Blocked edge** rendering (red dashed + X mark)
- **Control panel** with Reset/Step/Auto-play/Pause buttons
- **Live scoring** with color-coded value
- **Train status table** with position, delay, and status badges
- **Event log** showing conflicts and arrivals in real-time
- **Step history timeline** with delay/conflict bar chart
- **Session isolation** -- multiple concurrent users supported

## Docker

```bash
# Build the image
docker build -t railcascade-mini .

# Start the dashboard server
docker run -p 7860:7860 railcascade-mini

# Run inference agent inside Docker
docker run -e HF_TOKEN="your-key" railcascade-mini python inference.py medium
```

## Live Demo

Hugging Face Space: https://huggingface.co/spaces/irfanbasha/railcascade-mini
