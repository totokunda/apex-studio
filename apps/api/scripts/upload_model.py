import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from src.engine import UniversalEngine
path = "/home/tosin_coverquick_co/apex/manifest/verified/image/zimage-turbo-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=path).engine

repo_id = os.getenv("HF_REPO_ID")
if not repo_id:
    raise ValueError("HF_REPO_ID is not set")

engine.save_and_upload_component("scheduler", repo_id)