"""
This file can be used to wrap a 'regular' training file to run multiple
parameter combinations and save all outputs in a common folder.

"""
import os
import json
import copy
import time
import torch
import itertools
import numpy as np

import sys
sys.path.append('../../operator_learning/utils')
sys.path.append('../../operator_learning/training')
sys.path.append('../../operator_learning/models')
sys.path.append('../../operator_learning/dataloaders')
sys.path.append('../../operator_learning/datasets')

from parse_common_arguments import get_args


args = get_args()

assert args.settings is not None, "You must provide a settings file."

with open(args.settings) as f:
    settings = json.load(f)

run_file = settings.pop('run_file')
search_keys = settings.pop('grid_search_keys')
search_options = settings.pop('grid_search_options', dict())

# keys on the highest level may not be saved as a list
for kid in range(len(search_keys)):
    if not isinstance(search_keys[kid], list):
        search_keys[kid] = [search_keys[kid]]

# create settings files for grid search
options_list = []

for key_path in search_keys:
    options = settings

    for key_step in key_path:
        options = options[key_step]

    assert isinstance(options, list), \
        f"The value of option {key_path} is not a list."

    options_list.append(options)

options_list = list(itertools.product(*options_list))
n_options = 0
show_options = search_options.get('show_settings', False)

if show_options:
    print('Grid Search Settings:')

for eid, el in enumerate(options_list):
    if show_options:
        c_len = np.ceil(np.log10(len(options_list))).astype(int)
        print(f'  {eid:>{c_len}}: {el}')

    search_settings = copy.deepcopy(settings)

    for key_path, value in zip(search_keys, el):
        option = search_settings

        for key_step in key_path[:-1]:
            option = option[key_step]

        option[key_path[-1]] = value

    output_dir = os.path.join(args.output_dir, f"combination_{eid}")
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, "settings.json"), "w") as settings_file:
        json.dump(search_settings, settings_file, indent=4)

    n_options += 1

if show_options:
    print('')

# run jobs
skip_indices = search_options.get('skip_indices', list())
python_cmd = search_options.get('python_command', 'python')
job_prefix = search_options.get('job_prefix', '')

start_time = time.strftime(
    '%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

for eid in range(n_options):
    if eid in skip_indices:
        print(f'Skipping parameter combination {eid}.')
        continue

    if show_options:
        print(f'Running parameter combination {eid}.')

    output_dir = os.path.join(args.output_dir, f"combination_{eid}")
    settings_file = os.path.join(output_dir, "settings.json")

    cmd = (f"{job_prefix.format(time=start_time, eid=eid)} {python_cmd}"
           f" {run_file} --output_dir {output_dir}"
           f" --no_top_level_folder --no_output {settings_file}")
    print(cmd)
    os.system(cmd)
