import os
import sys
import json
import importlib

sys.path.append('utils')
from creator import SampleCreator
 
if len(sys.argv) != 2:
    raise ValueError('No settings file provided.')
  
settings_file = sys.argv[1]

assert os.path.exists(settings_file), \
    f"Settings file '{settings_file}' not found."

settings_dir = os.path.dirname(settings_file)
sys.path.append(settings_dir)

with open(settings_file) as f:
    settings = json.load(f)

functions_file_name = settings.get('functions_file', 'functions.py')
functions_file = os.path.join(settings_dir, functions_file_name)

assert os.path.exists(functions_file), \
    f"Functions file '{functions_file}' not found."

exec(f"import {functions_file_name.removesuffix('.py')} as functions")

problem_name = os.path.split(settings_dir)[1]
output_dir = os.path.join('created_data', problem_name)
os.makedirs(output_dir, exist_ok=True)

settings['output_file'] = os.path.join(output_dir, settings['output_file'])

creator = SampleCreator(
    settings,
    setup=functions.setup,
    get_input=functions.get_input,
    process_input=functions.process_input,
    solve=functions.solve,
    process_output=functions.process_output,
    finish=functions.finish)

creator.run()



