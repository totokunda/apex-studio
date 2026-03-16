from src.api.manifest import _load_and_enrich_manifest
import time
import glob 

paths = list(glob.glob("manifest/**/*.yml", recursive=True))   
print(f"Found {len(paths)} paths")

total_time = time.time()

for path in paths:
    t = time.time()
    content = _load_and_enrich_manifest(path.replace("manifest/", ""))
    end = time.time()
    print(f"Loaded and enriched manifest in {end - t} seconds")

total_time = time.time() - total_time
print(f"Total time: {total_time} seconds")