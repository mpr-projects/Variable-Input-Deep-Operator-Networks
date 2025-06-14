import h5py
import numpy as np

import torch
from torch.utils.data import TensorDataset

import normalizers
from standard_dataloaders import DataLoader


def convert_to_tuple(obj):
    try:
        obj = tuple(obj)
    except TypeError:
        obj = (obj,)

    return obj


def read_inputs_from_file(f, device):
    n_samples = f.attrs['n_samples']
    inputs = list()

    for sid in range(n_samples):
        inputs.append(torch.tensor(f[f'inputs/input_{sid}'][...], device=device))

    inputs = torch.stack(inputs, dim=0)
    return inputs.view(n_samples, -1, inputs.shape[-1])


def read_outputs_from_file(f, device):
    n_samples = f.attrs['n_samples']
    outputs = list()

    for sid in range(n_samples):
        outputs.append(torch.tensor(f[f'outputs/output_{sid}'][...], device=device))

    outputs = torch.stack(outputs, dim=0)
    return outputs.view(n_samples, -1, outputs.shape[-1])


def read_input_coords_from_file(f, device):
    input_coords = torch.tensor(f['inputs/input_coords'][...], device=device)
    dim = input_coords.shape[-1]
    return input_coords.view(-1, dim)


def read_output_coords_from_file(f, device):
    output_coords = torch.tensor(f['outputs/output_coords'][...], device=device)
    dim = output_coords.shape[-1]
    return output_coords.view(-1, dim)


class InputOutputDatasetWithoutPreloading(torch.utils.data.Dataset):
    """
    Creates a dataset with both inputs and outputs. Data is loaded from a
    hdf5-file when needed. This reduces the required memory but slows
    down access time.

    """
    def __init__(self, file, device=None):
        self.f = h5py.File(file, 'r')
        self.device = device
        self.n_samples = int(self.f.attrs['n_samples'])

    def __del__(self):
        try:
            self.close()
        except TypeError:
            pass

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, sid):
        inputs = torch.tensor(
            self.f[f'inputs/input_{sid}'][...], device=self.device)

        outputs = torch.tensor(
            self.f[f'outputs/output_{sid}'][...], device=self.device)

        return inputs, outputs


def CreateDeepONetDataloader(file, batch_size=None, shuffle=False,
                             inputs_hook=[], outputs_hook=[],
                             output_coords_hook=[],
                             n_out_coords_per_epoch=None,
                             device=None):
    """
    Return a dataloader which can be used for a regular DeepONet.
    'file' must be a hdf5-file containing the input and output data.

    """
    f = h5py.File(file, 'r')

    inputs = read_inputs_from_file(f, device)
    outputs = read_outputs_from_file(f, device)
    output_coords = read_output_coords_from_file(f, device)

    inputs_hook = convert_to_tuple(inputs_hook)
    outputs_hook = convert_to_tuple(outputs_hook)
    output_coords_hook = convert_to_tuple(output_coords_hook)

    for hook in inputs_hook:
        inputs = hook(inputs, f)

    for hook in outputs_hook:
        outputs = hook(outputs, f)

    for hook in output_coords_hook:
        output_coords = hook(output_coords, f)

    dataset = TensorDataset(inputs, outputs)

    if batch_size is not None:
        batch_size = min(batch_size, len(inputs))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    # if requested, only use a subset of all output coordinates each epoch
    n_out_coords = len(output_coords)
    n_out_coords_per_epoch = n_out_coords_per_epoch or n_out_coords

    if n_out_coords_per_epoch > n_out_coords:
        print(f"Warning: 'n_out_coords_per_epoch' ({n_out_coords_per_epoch})"
              " is larger than the total number of coordinates"
              f" ({n_out_coords}). Using the number of coordinates instead.")
        n_out_coords_per_epoch = n_out_coords 

    def reshape_fn(data):
        inputs, targets = data
        indices = torch.randperm(targets.shape[1])[:n_out_coords_per_epoch]
        model_input = (inputs, output_coords[indices]) 
        return model_input, targets[:, indices]

    f.close()
    return dataloader, reshape_fn


def CreateDeepONetDataloaderWithoutPreloading(file, batch_size=None,
                                              shuffle=False,
                                              inputs_hook=[], outputs_hook=[],
                                              output_coords_hook=[],
                                              device=None):
    """
    Equivalent to 'CreateDeepONetDataloader' but data is loaded when needed.

    """
    f = h5py.File(file, 'r')

    dataset = InputOutputDatasetWithoutPreloading(file, device=device)
    output_coords = read_output_coords_from_file(f, device)

    if batch_size is not None:
        batch_size = min(batch_size, len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    inputs_hook = convert_to_tuple(inputs_hook)
    outputs_hook = convert_to_tuple(outputs_hook)
    output_coords_hook = convert_to_tuple(output_coords_hook)

    for hook in output_coords_hook:
        output_coords = hook(output_coords, f)

    def reshape_fn(data):
        inputs, targets = data

        for hook in inputs_hook:
            inputs = hook(inputs, f)

        for hook in outputs_hook:
            targets = hook(targets, f)

        targets = targets.view(targets.shape[0], -1, targets.shape[-1])

        model_input = (inputs, output_coords) 
        return model_input, targets

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
