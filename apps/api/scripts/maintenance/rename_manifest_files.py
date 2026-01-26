import os
from glob import glob
from src.utils.yaml import load_yaml

path = "manifest"

yaml_files = glob(f"{path}/**/*.yml", recursive=True)


for yaml_file in yaml_files:
    if "shared" in yaml_file or "legacy" in yaml_file:
        continue

    data = load_yaml(yaml_file)

    dir_name = os.path.dirname(yaml_file)
    name = data["metadata"]["name"].lower().replace(" ", "-")
    version = data["metadata"]["version"]

    full_name = f"{name}-{version}.v1.yml"

    full_path = os.path.join(dir_name, full_name)

    print("Renaming", yaml_file, "to", full_path)

    os.rename(yaml_file, full_path)
