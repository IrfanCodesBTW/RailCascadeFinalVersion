import os
import uvicorn
from inference import app

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()