---
title: RailCascade OpenEnv
emoji: 🚆
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RailCascadeMiniV2

**Deterministic Railway Traffic Management RL Environment**

A 10-node directed graph network where an AI agent must coordinate multiple trains, issuing per-train hold/reroute/noop commands each timestep to minimize total delay caused by blocked edges, edge conflicts, and node-occupancy cascading.

# RailCascade OpenEnv Environment

RailCascade is a real-world inspired railway traffic control simulation built using the OpenEnv standard.

## Features

- Multi-train routing with conflict resolution
- Cascading delays and blockage simulation
- Deterministic policy agent (no external API required)
- Step-based environment (`step()`, `reset()`, `state()`)

## API Endpoints

- `/reset` → Reset environment
- `/step` → Execute action
- `/state` → Get current state

## Deployment

This Space runs using a Docker container and exposes a FastAPI backend.

## Notes

- Uses a deterministic policy agent (no OpenAI API calls required)
- Fully compliant with OpenEnv hackathon requirements