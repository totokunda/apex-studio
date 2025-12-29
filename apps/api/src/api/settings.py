import os
from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    results_dir: Path = Path(os.getenv("RESULTS_DIR", Path.home() / ".apex/results"))
    max_jobs_per_gpu: int = int(os.getenv("MAX_JOBS_PER_GPU", "1"))
    ray_address: str = os.getenv("RAY_ADDRESS", "auto")  # "auto" for local mode
    ray_dashboard_port: int = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))


settings = Settings()
