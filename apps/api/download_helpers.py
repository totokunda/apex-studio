import yaml 
from pprint import pprint
from src.mixins.download_mixin import DownloadMixin
import os 

os.makedirs('helpers', exist_ok=True)
dl = DownloadMixin()


with open('helpers.yml', 'r') as f:
    data = yaml.safe_load(f)

for helper in data:
    
    model_path = helper.get('model_path', [])
    if not model_path:
        continue
    path = model_path[0]['path']
    dl.download(path, os.path.join('helpers', path))