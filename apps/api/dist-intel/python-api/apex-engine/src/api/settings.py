import os
from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    results_dir: Path = Path(os.getenv("RESULTS_DIR", Path.home() / ".apex/results"))
    max_jobs_per_gpu: int = int(os.getenv("MAX_JOBS_PER_GPU", "1"))
    # IMPORTANT:
    # - Ray itself uses the env var `RAY_ADDRESS` to auto-connect to a cluster.
    # - For this API, we want to start a *local* Ray instance by default.
    #   Use `APEX_RAY_ADDRESS` explicitly to connect to a remote cluster ("auto" or an address).
    ray_address: str = os.getenv("APEX_RAY_ADDRESS", "local")
    ray_dashboard_port: int = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))


settings = Settings()
