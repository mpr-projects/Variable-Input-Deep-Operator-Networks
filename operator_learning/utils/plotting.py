import os
import sys
import h5py
import json
import argparse
import matplotlib.pyplot as plt


# Applicable to SCALAR problem outputs only!

def parse_plotting_arguments():
    parser = argparse.ArgumentParser(description='Plot results of DeepONets and TDeepONets.')

    parser.add_argument('output_folder')

    parser.add_argument('--settings_file', default=None, help=(
        "Name of settings (.json) file to use, must be present in 'output_folder'."
        " If not provided then the first .json file found will be used."))

    parser.add_argument('--source_file', default='test_file', help=(
        "Must be a key available in the settings (.json) file available in"
        " 'output_folder' whose value is the path to the source of the dataset to use."
        " Defaults to 'test_file'."))

    parser.add_argument('--model_file', default=None, help=(
        "Name of of model (.pt) file saved during the training process in the"
        " checkpoint subfolder of 'output_folder'. If not provided then the first"
        " .pt file found will be used."))

    args = parser.parse_args()

    if args.settings_file is None:
        files = os.listdir(args.output_folder)
        files = [file for file in files if file[-5:] == '.json']
        args.settings_file = files[0]

    if os.path.split(args.settings_file)[0] == '':
        args.settings_file = os.path.join(args.output_folder, args.settings_file)

    assert os.path.exists(args.settings_file), \
        f"Couldn't find settings file {args.settings_file}."

    if args.model_file is None:
        files = os.listdir(os.path.join(args.output_folder, 'checkpoints'))
        files = [file for file in files if file[-3:] == '.pt']
        args.model_file = files[0]

    if os.path.split(args.model_file)[0] == '':
        args.model_file = os.path.join(args.output_folder, 'checkpoints', args.model_file)

    assert os.path.exists(args.model_file), \
        f"Couldn't find model file {args.model_file}."

    with open(args.settings_file) as f:
        settings = json.load(f)

    assert args.source_file in settings, \
        f"Key '{args.source_file}' not found in settings file {args.settings_file}."

    return args


def plot_1d(title, coords, values, time_dependent, figTuple):
    fig, tup = figTuple
    ax = fig.add_subplot(*tup)

    p = ax.scatter(coords[:, 0], values, c=values)
    plt.colorbar(p, ax=ax, location='bottom')
    ax.set_title(title)
    ax.set_xlabel('t' if time_dependent else 'x')


def plot_2d_time(title, coords, values, figTuple):
    fig, tup = figTuple
    ax = fig.add_subplot(*tup)

    p = ax.scatter(coords[:, 1], coords[:, 0], c=values, s=1.8)
    plt.colorbar(p, ax=ax, location='bottom')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('t')


def plot_2d_steady_state(title, coords, values, figTuple):
    fig, tup = figTuple
    ax = fig.add_subplot(*tup)

    p = ax.scatter(coords[:, 1], coords[:, 0], c=values, s=5)
    plt.colorbar(p, ax=ax, location='bottom')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_2d(title, coords, values, time_dependent, figTuple):
    if time_dependent:
        plot_2d_time(title, coords, values, figTuple)

    else:
        plot_2d_steady_state(title, coords, values, figTuple)


def plot_3d_time(title, coords, values, figTuple):
    fig, tup = figTuple
    ax = fig.add_subplot(*tup, projection='3d')

    p = ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0], c=values)
    plt.colorbar(p, ax=ax, location='bottom')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')


def plot_3d_steady_state(title, coords, values, figTuple):
    fig, tup = figTuple
    ax = fig.add_subplot(*tup, projection='3d')

    p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values)
    plt.colorbar(p, ax=ax, location='bottom')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_3d(title, coords, values, time_dependent, figTuple):
    if time_dependent:
        plot_3d_time(title, coords, values, figTuple)

    else:
        plot_3d_steady_state(title, coords, values, figTuple)


plot_fns = {
    1: plot_1d,
    2: plot_2d,
    3: plot_3d
}


def plot_sample(sid, sensor_values, preds, targets, input_coords, output_coords, time_dependent_input, time_dependent_output):

    # find correct functions for plotting
    dim_in_coords = input_coords.shape[-1]
    
    assert dim_in_coords in [1, 2], \
        "Currently only one- or two-dimensional sensor coordinates are supported."

    dim_inputs = sensor_values.shape[-1]

    dim_out_coords = output_coords.shape[-1]

    assert dim_out_coords in [1, 2, 3], \
        "Only one-, two- or three-dimensional output coordinates are supported."

    # plot data
    fig = plt.figure()

    ncols = dim_inputs + 1
    pid = 1
    
    for idx in range(dim_inputs):
        figTuple = (fig, (2, ncols, pid))
        plot_fns[dim_in_coords](
            'Sensor Values' + (f' {idx}' if dim_inputs > 1 else ''),
            input_coords, sensor_values[..., idx], time_dependent_input, figTuple)
        pid += 1

    output_coords = output_coords.reshape(-1, output_coords.shape[-1])

    figTuple = (fig, (2, ncols, pid))
    plot_fns[dim_out_coords]('Predictions', output_coords, preds, time_dependent_output, figTuple)
    pid += 1

    diff = preds - targets

    figTuple = (fig, (2, ncols, pid))
    plot_fns[dim_out_coords]('Differences', output_coords, diff, time_dependent_output, figTuple)
    pid += 1

    figTuple = (fig, (2, ncols, pid))
    plot_fns[dim_out_coords]('Targets', output_coords, targets, time_dependent_output, figTuple)

    l2_error = ((diff**2).sum() / (targets**2).sum())**0.5

    fig.suptitle(f'Sample {sid} - Error {l2_error*100:.2f}%')
    plt.show()


def plot_samples(model, dataloader, input_coords, output_coords,
         time_dependent_input, time_dependent_output,
         reshape_fn, output_transform, inputs_reshape_fn, outputs_reshape_fn):
    """
    The 'dataloader' must have batch_size set to 1. 'inputs_reshape_fn' is
    applied to sensor inputs before plotting. Those inputs can include more
    data than just the sensor values, the reshape function should return
    just the sensor values.
    
    To consider: inputs[1] contains output_coords (for versions of the DeepONet),
                 is there a point in asking for output_coords separately?

    """
    for sid, data in enumerate(dataloader):
        inputs, targets = reshape_fn(data)

        assert len(targets) == 1, \
            f"'dataloader' must have batch size set to 1 (it's set to {len(targets)} instead)."

        preds = output_transform(inputs, model(inputs)).detach()

        input_coords, sensor_values = inputs_reshape_fn(input_coords, inputs)
        output_coords, preds, targets = outputs_reshape_fn(
            output_coords, inputs, preds, targets)

        preds = preds.view(-1)
        targets = targets.view(-1)

        plot_sample(sid, sensor_values, preds, targets,
                    input_coords, output_coords,
                    time_dependent_input, time_dependent_output)
