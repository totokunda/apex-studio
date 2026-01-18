import os 
from glob import glob
import shutil
path = '/Users/tosinkuye/Library/Application Support/Apex Studio/media/6ad073e5-7b45-43b8-aeab-bab0782d3eb7' 
server_path = os.path.join(path, 'server')
os.makedirs(server_path, exist_ok=True)
# make dirs generation_results and processor_results
generation_results_path = os.path.join(server_path, 'generations')
processor_results_path = os.path.join(server_path, 'processors')
os.makedirs(generation_results_path, exist_ok=True)
os.makedirs(processor_results_path, exist_ok=True)

c_remote = '/Users/tosinkuye/Library/Application Support/apex-studio/apex-cache-remote'

# get all directories in c_remote
dirs = glob(os.path.join(c_remote, '*'))
for dir in dirs:
    # get the name of the directory
    name = os.path.basename(dir)
    # move to server_path
    print(f"Moving {name} to {os.path.join(processor_results_path, name)}")
    shutil.move(dir, os.path.join(processor_results_path, name))