import yaml 
from glob import glob
import pydash as _
from pprint import pprint
files = glob('new_manifest/**/*.yml', recursive=True)

helpers = []

for file in files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        components = _.get(data, 'spec.components', [])
        
        for component in components:
            if component.get('type') == 'helper':
                helpers.append(component)

with open('helpers.yml', 'w') as f:
    yaml.dump(helpers, f)