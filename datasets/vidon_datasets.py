import sys
import h5py
import torch

from torch.nn.utils.rnn import pad_sequence


class VidonDataset(torch.utils.data.Dataset):
    def __init__(self, source_file_name, inputs_hook=[], outputs_hook=[], device=None):
        super().__init__()

        f = h5py.File(source_file_name)
        self.n_samples = f.attrs['n_samples']

        self.inputs = list()
        self.input_coord_ids = torch.zeros(self.n_samples, dtype=int)
        self.input_coords_inds = list()

        self.outputs = list()
        self.output_coord_ids = torch.zeros(self.n_samples, dtype=int)
        self.output_coords_inds = list()

        self.input_coords = torch.tensor(f['inputs/input_coords'][...], device=device)

        for sid in range(self.n_samples):
            data = torch.tensor(f[f'inputs/input_{sid}'][...], device=device)

            for hook in inputs_hook:
                data = hook(data, f)

            self.inputs.append(data)
            self.input_coord_ids[sid] = f[f'inputs/input_{sid}'].attrs['coord_id']

        n_input_coord_ids = self.input_coord_ids.max().item() + 1

        for cid in range(n_input_coord_ids):
            self.input_coords_inds.append(
                torch.tensor(f[f'inputs/input_coords_{cid}'][...].astype(int), device=device))

        self.output_coords = torch.tensor(f['outputs/output_coords'][...], device=device)

        for sid in range(self.n_samples):
            data = torch.tensor(f[f'outputs/output_{sid}'][...], device=device)

            for hook in outputs_hook:
                data = hook(data, f)

            self.outputs.append(data)
            self.output_coord_ids[sid] = f[f'outputs/output_{sid}'].attrs['coord_id']

        n_output_coord_ids = self.output_coord_ids.max().item() + 1

        for cid in range(n_output_coord_ids):
            self.output_coords_inds.append(
                torch.tensor(f[f'outputs/output_coords_{cid}'][...].astype(int), device=device))

        f.close()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, sid):
        return (self.inputs[sid],
                self.input_coords_inds[self.input_coord_ids[sid]],
                self.outputs[sid],
                self.output_coords_inds[self.output_coord_ids[sid]])



def vidon_collate(data):
    # data is a list with one entry for each sample
    inputs = [d[0] for d in data]
    input_coord_inds = [d[1] for d in data]
    outputs = [d[2] for d in data]
    output_coord_inds = [d[3] for d in data]

    inputs = pad_sequence(inputs, batch_first=True)
    input_coord_inds = pad_sequence(input_coord_inds, batch_first=True, padding_value=-1)
    outputs = pad_sequence(outputs, batch_first=True)
    output_coord_inds = pad_sequence(output_coord_inds, batch_first=True, padding_value=-1)

    return input_coord_inds, inputs, output_coord_inds, outputs
