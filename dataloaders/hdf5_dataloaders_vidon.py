import h5py
import numpy as np

import torch
from torch.utils.data import TensorDataset

from standard_dataloaders import DataLoader

import vidon_datasets
import normalizers


def convert_to_tuple(obj):
    try:
        obj = tuple(obj)
    except TypeError:
        obj = (obj,)

    return obj


def read_inputs_from_file(f, device, pad_value=0):
    n_samples = f.attrs['n_samples']

    inputs = list()

    for sid in range(n_samples):
        inputs.append(torch.tensor(f[f'inputs/input_{sid}'][...], device=device))

    if pad_value is None:
        return inputs

    inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=pad_value)

    return inputs.view(n_samples, -1, inputs.shape[-1])


def read_outputs_from_file(f, device, pad_value=0):
    n_samples = f.attrs['n_samples']

    outputs = list()

    for sid in range(n_samples):
        outputs.append(torch.tensor(f[f'outputs/output_{sid}'][...], device=device))

    if pad_value is None:
        return outputs

    outputs = torch.nn.utils.rnn.pad_sequence(
        outputs, batch_first=True, padding_value=pad_value)

    return outputs.view(n_samples, -1, outputs.shape[-1])


def read_input_coords_from_file(f, device):
    input_coords = torch.tensor(f['inputs/input_coords'][...], device=device)

    dim = input_coords.shape[-1]
    return input_coords.view(-1, dim)


def read_output_coords_from_file(f, device):
    output_coords = torch.tensor(f['outputs/output_coords'][...], device=device)

    dim = output_coords.shape[-1]
    return output_coords.view(-1, dim)


def CreateVidonDataloader(file, batch_size=None,
                              inputs_hook=[], outputs_hook=[],
                              n_out_coords_per_epoch=None,
                              shuffle=False, device=None):
    """
    Dataloader for source files converted to Vidon-specific format.

    """
    inputs_hook = convert_to_tuple(inputs_hook)
    outputs_hook = convert_to_tuple(outputs_hook)

    dataset = vidon_datasets.VidonDataset(
        file, inputs_hook=inputs_hook, outputs_hook=outputs_hook, device=device)

    if batch_size is not None:
        batch_size = min(batch_size, len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=True, collate_fn=vidon_datasets.vidon_collate)

    # if requested, only use a subset of all output coordinates in every epoch
    n_out_coords = len(dataset.output_coords)
    n_out_coords_per_epoch = n_out_coords_per_epoch or n_out_coords

    if n_out_coords_per_epoch > n_out_coords:
        print(f"Warning: 'n_out_coords_per_epoch' ({n_out_coords_per_epoch})"
              " is larger than the total number of coordinates"
              f" ({n_out_coords}). Using the number of coordinates instead.")
        n_out_coords_per_epoch = n_out_coords 

    def reshape_fn(data):
        in_coord_inds, inputs, out_coord_inds, outputs = data
        indices = torch.randperm(outputs.shape[1])[:n_out_coords_per_epoch]

        model_input = (
            dataset.input_coords,
            in_coord_inds,
            inputs,
            dataset.output_coords,
            out_coord_inds[:, indices]) 

        return model_input, outputs[:, indices]

    return dataloader, reshape_fn


def CreateVidonDataloaderDefaultFormat(file, batch_size=None,
                                           shuffle=False, device=None,
                                           inputs_hook=[], input_coords_hook=[],
                                           outputs_hook=[], output_coords_hook=[],
                                           n_out_coords_per_epoch=None,
                                           return_output_coords=False):
    """
    Dataloader for default file format, i.e. files that were not converted
    to the Vidon-specific format. This dataloader takes the entire
    source file without subsampling (if you want subsampling then pre-process
    the data).

    """
    f = h5py.File(file, 'r')

    inputs = read_inputs_from_file(f, device)
    outputs = read_outputs_from_file(f, device)
    input_coords = read_input_coords_from_file(f, device)
    output_coords = read_output_coords_from_file(f, device)

    inputs_hook = convert_to_tuple(inputs_hook)
    outputs_hook = convert_to_tuple(outputs_hook)
    input_coords_hook = convert_to_tuple(input_coords_hook)
    output_coords_hook = convert_to_tuple(output_coords_hook)

    for hook in inputs_hook:
        inputs = hook(inputs, f)

    for hook in outputs_hook:
        outputs = hook(outputs, f)

    for hook in input_coords_hook:
        input_coords = hook(input_coords, f)

    for hook in output_coords_hook:
        output_coords = hook(output_coords, f)

    inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
    dataset = TensorDataset(inputs, outputs)

    if batch_size is not None:
        batch_size = min(batch_size, len(inputs))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    in_coord_inds = torch.arange(len(input_coords), device=device).view(1, -1, 1)
    out_coord_inds = torch.arange(len(output_coords), device=device).view(1, -1, 1)

    def reshape_fn(data):
        inputs, targets = data
        n_samples = len(inputs)
        inds_in = torch.repeat_interleave(in_coord_inds, n_samples, dim=0)
        inds_out = torch.repeat_interleave(out_coord_inds, n_samples, dim=0)
        model_input = (input_coords, inds_in, inputs, output_coords, inds_out) 
        return model_input, targets

    f.close()

    if return_output_coords:
        return dataloader, reshape_fn, output_coords

    return dataloader, reshape_fn


# ----------------------------------------------------------------------------
# Create Normalizers on Data
# ----------------------------------------------------------------------------
def get_normalizer(source_file, normalizer_name, which, device, pad_value=0):
    """
    Loads data in source_file and initializes 'normalizer_name' with it.
    If 'normalizer_name' is None then dummy functions will be returned.
    Argument 'which' must be either 'input' or 'output'.

    """
    if normalizer_name is None:
        normalizer = lambda data, file: data
        output_transform = lambda inputs, outputs: outputs 
        return normalizer, output_transform

    assert which in ['input', 'output'], \
        "Argument 'which' must be either 'input' or 'output'."

    read_fns = {
        'input': read_inputs_from_file,
        'output': read_outputs_from_file}

    with h5py.File(source_file) as f:
        data = read_fns[which](f, device, pad_value=None)

    data = torch.cat(data)
    normalizer = getattr(normalizers, normalizer_name)(data)

    def output_transform(inputs, outputs):
        outputs = normalizer.invert(outputs)
        pad_inds = (inputs[4] == -1)
        outputs[pad_inds] = 0
        return outputs

    return normalizer, output_transform
