import os
import sys
import copy
import json
import h5py
import torch
import argparse
import numpy as np

sys.path.append('../../operator_learning/utils')
sys.path.append('../../operator_learning/training')
sys.path.append('../../operator_learning/models')
sys.path.append('../../operator_learning/dataloaders')
sys.path.append('../../operator_learning/datasets')

from plotting import parse_plotting_arguments, plot_samples
from error_calculation import compute_error, relative_Lp_error

# if there are new models you need to update the mappings file
from compute_model_errors_dataloader_mappings import *


def _compute(output_folder, json_folder=None, print_output=True, model_file=None, remove_existing_results=True):
    # .json file may not be in the same folder as the other outputs (eg retrainer)
    json_folder = json_folder or output_folder

    # if there are multiple .json or .pt files then the first one in the
    # list will be chosen
    settings_file = [f for f in os.listdir(json_folder) if f[-5:] == '.json']
    settings_file.sort()
    settings_file = os.path.join(json_folder, settings_file[0])

    with open(settings_file) as f:
        settings = json.load(f)

    training_file = settings['training_file']
    validation_file = settings['validation_file']
    test_file = settings['test_file']

    # if no model file was provided then the first one in folder 'checkpoints'
    # will be used
    if model_file is None:
        model_folder = os.path.join(output_folder, 'checkpoints')
        model_file = [f for f in os.listdir(model_folder) if f[-3:] == '.pt']
        model_file.sort()
        print('Using model file:', model_file[0])
        model_file = os.path.join(model_folder, model_file[0])

    saved_dict = torch.load(model_file, map_location='cpu')

    # get normalizers and other hooks
    get_normalizer = get_normalizer_mapping[saved_dict['model_creation_fn']]

    normalizer_inputs, _ = get_normalizer(
        training_file, settings.get('normalizer_inputs', None), 'input', device=None)

    _, output_transform = get_normalizer(
        training_file, settings.get('normalizer_outputs', None), 'output', device=None)

    inputs_hook = (normalizer_inputs,)
    inputs_hook_post, outputs_hook = tuple(), tuple()
    in_coords_hook, out_coords_hook = tuple(), tuple()

    # compute errors
    results_file = os.path.join(output_folder, 'errors.txt')

    if os.path.exists(results_file) and remove_existing_results:
        os.remove(results_file)

    def save_result(which, result):
        if print_output:
            print(f'\nRelative L2-error ({which}): {result*100:.3f}%')

        with open(results_file, 'a') as f:
            f.write(f'Relative L2-error ({which}): {result*100:.3f}%\n')

    files = {
        "training": training_file,
        "validation": validation_file,
        "test": test_file
    }

    for name, data_file in files.items():
        get_dl = dataloader_mapping[saved_dict['model_creation_fn']]
        model, (dataloader, reshape_fn) = get_dl(
            name, model_file, data_file, settings, inputs_hook)

        rel_l2_error = compute_error(
            model,
            dataloader,
            relative_Lp_error(sample_dimensions=0, batch_dimension=0),
            reshape_fn=reshape_fn,
            output_transform=output_transform)

        mean_rel_l2_error = rel_l2_error.mean()
        save_result(name, mean_rel_l2_error)

    with open(results_file, 'a') as f:
        f.write('\n')


def _compute_run_statistics(output_folder, combs):
    # get labels
    f = os.path.join(output_folder, combs[0], 'errors.txt')

    with open(f) as f:
        contents = f.read()

    contents = contents.strip().split('\n')
    names = list()

    for c in contents:
        names.append(c.split(':')[0])

    errors = np.zeros((len(names), len(combs)))

    # read in results
    for cid, comb in enumerate(combs):
        f = os.path.join(output_folder, comb, 'errors.txt')

        with open(f) as f:
            contents = f.read()

        contents = contents.strip().split('\n')

        for rid, c in enumerate(contents):
            errors[rid, cid] = float(c.split(':')[1][:-1])

    mean_errors = np.mean(errors, axis=1)
    std_errors = np.std(errors, axis=1)

    # save results
    results_file = os.path.join(output_folder, 'results_statistics.txt')

    with open(results_file, 'w') as f:
        for nid, name in enumerate(names):
            f.write(f'{name}: {mean_errors[nid]:.2f}% +/- {std_errors[nid]:.2f}%\n')


def _compute_gs_statistics(output_folder, combs):
    # get labels
    f = os.path.join(output_folder, combs[0], 'results_statistics.txt')

    with open(f) as f:
        contents = f.read()

    contents = contents.strip().split('\n')
    names = list()
    errors = list()

    for c in contents:
        names.append(c.split(':')[0])
        errors.append(list())

    # read in results
    for cid, comb in enumerate(combs):
        f = os.path.join(output_folder, comb, 'results_statistics.txt')

        with open(f) as f:
            contents = f.read()

        contents = contents.strip().split('\n')

        for rid, c in enumerate(contents):
            mean_std = c.split(':')[1]
            errors[rid].append(mean_std)

    # save results
    results_file = os.path.join(output_folder, 'results_statistics.txt')
    c_len = np.ceil(np.log10(len(combs))).astype(int)

    with open(results_file, 'w') as f:
        for nid, name in enumerate(names):
            f.write(f'{name}:\n')
            for eid, err in enumerate(errors[nid]):
                f.write(f'  {eid:>{c_len}}:{err}\n')
            f.write('\n')


def compute(args):
    args = copy.deepcopy(args)
    contents = os.listdir(args.output_folder)

    # check if output is from grid search
    combs = [f for f in contents if f[:12] == 'combination_']

    if len(combs) > 0:
        output_folder = args.output_folder

        for comb in combs:
            args.output_folder = os.path.join(output_folder, comb)
            compute(args)

        _compute_gs_statistics(output_folder, combs)
        return

    # check if output is from retrainer
    combs = [f for f in contents if f[:4] == 'run_']

    if len(combs) > 0:
        output_folder = args.output_folder

        for comb in combs:
            args.output_folder = os.path.join(output_folder, comb)
            args.json_folder = output_folder
            compute(args)

        _compute_run_statistics(output_folder, combs)
        return

    # compute error
    _compute(args.output_folder,
             json_folder=args.json_folder or args.output_folder,
             print_output=not args.no_output,
             model_file=args.model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute errors of saved models.')
    parser.add_argument('output_folder')
    parser.add_argument('--json_folder', default=None)
    parser.add_argument('--model_file', default=None)
    parser.add_argument('--no_output', action='store_true')

    args = parser.parse_args()
    compute(args)
