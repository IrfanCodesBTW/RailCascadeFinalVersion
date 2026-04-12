# This file is intentionally a no-op entrypoint.
# The Dockerfile CMD uses uvicorn CLI directly.
# Do NOT call uvicorn.run() here — it causes double-bind on port 7860.

from inference import app  # noqa: F401 — re-exported for any external import

