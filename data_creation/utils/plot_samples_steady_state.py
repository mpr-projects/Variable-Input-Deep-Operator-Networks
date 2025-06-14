import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


assert len(sys.argv) == 2, f"You must provide a .hdf5-file name when calling {sys.argv[0]}."


def plot_sample(sid, f, show_title=True):
    print(f'Plotting sample {sid}:')

    inputs = f[f'inputs/input_{sid}'][...]
    outputs = f[f'outputs/output_{sid}'][...]

    input_dim, output_dim = inputs.shape[-1], outputs.shape[-1]

    input_coords = f[f'inputs/input_coords'][...]

    try:
        ic_sid = f[f'inputs/input_coords_{sid}'][...].reshape(-1).astype(int)
        input_coords = input_coords[ic_sid]

    except KeyError:
        pass

    fig, axs = plt.subplots(ncols=input_dim + output_dim, figsize=(9, 3))
    ax_id = 0

    for idx in range(input_dim):
        print('  Number of input coordinates:', len(input_coords))
        p = axs[ax_id].scatter(input_coords[..., 0], input_coords[..., 1], c=inputs[..., idx], s=1)
        plt.colorbar(p, ax=axs[ax_id])
        axs[ax_id].set_title('Input' + (f' {idx}' if input_dim > 1 else ''))
        axs[ax_id].set_xlabel('x')
        axs[ax_id].set_ylabel('y')
        axs[ax_id].set_aspect('equal', 'box')
        ax_id += 1

    output_coords = f[f'outputs/output_coords'][...]

    try:
        oc_sid = f[f'outputs/output_coords_{sid}'][...].reshape(-1).astype(int)
        output_coords = output_coords[oc_sid]

    except KeyError:
        pass


    for idx in range(output_dim):
        print('  Number of output coordinates:', len(output_coords))
        p = axs[ax_id].scatter(output_coords[..., 0], output_coords[..., 1], c=outputs[..., idx], s=1)
        plt.colorbar(p, ax=axs[ax_id])  # , location='bottom')
        axs[ax_id].set_title('Output' + (f' {idx}' if output_dim > 1 else ''))
        axs[ax_id].set_xlabel('x')
        axs[ax_id].set_ylabel('y')
        axs[ax_id].set_aspect('equal', 'box')
        ax_id += 1

    if show_title:
        fig.suptitle(f'Sample {sid}')

    plt.show()


with h5py.File(sys.argv[1], 'r') as source_file:
    n_samples = source_file.attrs['n_samples']

    print(f"The dataset contains {n_samples} samples. A scatter plot is used."
          " It can be difficult to see the sample structure if the plot is too"
          " large. Reduce the size of the plot if necessary.\n\n"
          "Press 'q' to show the next sample, press Ctrl-C to cancel.")

    for sid in range(n_samples):
        plot_sample(sid, source_file, show_title=True)
