import h5py
import numpy as np

import torch
from torch.utils.data import TensorDataset

import normalizers
from standard_dataloaders import DataLoader


def read_inputs_from_file(f, device):
    n_samples = f.attrs['n_samples']
    inputs = list()

    for sid in range(n_samples):
        inputs.append(torch.tensor(f[f'inputs/input_{sid}'][...], device=device))

    return torch.stack(inputs, dim=0)


def read_outputs_from_file(f, device):
    n_samples = f.attrs['n_samples']
    outputs = list()

    for sid in range(n_samples):
        outputs.append(torch.tensor(f[f'outputs/output_{sid}'][...], device=device))

    return torch.stack(outputs, dim=0)


def read_input_coords_from_file(f, device):
    input_coords = torch.tensor(f['inputs/input_coords'][...], device=device)
    dim = input_coords.shape[-1]
    return input_coords.view(-1, dim)


def read_output_coords_from_file(f, device):
    output_coords = torch.tensor(f['outputs/output_coords'][...], device=device)
    dim = output_coords.shape[-1]
    return output_coords.view(-1, dim)


def CreateFNODataloader(file, batch_size=None, shuffle=False,
                             inputs_hook=[], outputs_hook=[],
                             output_coords_hook=[],
                             n_out_coords_per_epoch=None,
                             device=None):
    """
    This function returns a dataloader which can be used for a regular
    DeepONet. 'file' must be a hdf5-file containing the input and
    output data.

    """
    f = h5py.File(file, 'r')

    inputs = read_inputs_from_file(f, device)
    outputs = read_outputs_from_file(f, device)

    try:
        inputs_hook = tuple(inputs_hook)
    except TypeError:
        inputs_hook = (inputs_hook,)

    try:
        outputs_hook = tuple(outputs_hook)
    except TypeError:
        outputs_hook = (outputs_hook,)

    for hook in inputs_hook:
        inputs = hook(inputs, f)

    for hook in outputs_hook:
        outputs = hook(outputs, f)

    dataset = TensorDataset(inputs, outputs)

    if batch_size is not None:
        batch_size = min(batch_size, len(inputs))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def reshape_fn(data):
        return data

    f.close()
    return dataloader, reshape_fn


# ----------------------------------------------------------------------------
# Create Normalizers on Data
# ----------------------------------------------------------------------------
def get_normalizer(source_file, normalizer_name, which, device):
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
        data = read_fns[which](f, device)

    normalizer = getattr(normalizers, normalizer_name)(data)
    output_transform = lambda inputs, outputs: normalizer.invert(outputs)
    return normalizer, output_transform
