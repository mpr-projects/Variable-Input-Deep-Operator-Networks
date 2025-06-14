import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


assert len(sys.argv) == 2, f"You must provide the file name when calling {sys.argv[0]}."


def plot_sample(sid, f, show_title=True):
    print(f'Plotting sample {sid}:')

    inputs = f[f'inputs/input_{sid}'][...]
    input_coords = f['inputs/input_coords'][...]
    input_coords = input_coords.reshape(-1, input_coords.shape[-1])
    print('  Number of input coordinates:', len(input_coords))

    try:
        ic_sid = f[f'inputs/input_coords_{sid}'][...].reshape(-1).astype(int)
        input_coords = input_coords[ic_sid]

    except KeyError:
        pass

    fig = plt.figure(figsize=(9, 3))
    ax0 = fig.add_subplot(1, 2, 1)

    p = ax0.scatter(input_coords[..., 0], input_coords[..., 1], c=inputs[..., 0], s=1)
    plt.colorbar(p, ax=ax0)  # , location='bottom')
    ax0.set_title('Inputs')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    outputs = f[f'outputs/output_{sid}'][...]
    outputs = outputs.reshape(-1, outputs.shape[-1])

    output_coords = f[f'outputs/output_coords'][...]
    output_coords = output_coords.reshape(-1, output_coords.shape[-1])
    print('  Number of output coordinates:', len(output_coords))

    try:
        oc_sid = f[f'outputs/output_coords_{sid}'][...].reshape(-1).astype(int)
        output_coords = output_coords[oc_sid]

    except KeyError:
        pass

    p = ax1.scatter(output_coords[:, 2], output_coords[:, 1], output_coords[:, 0], c=outputs[:, 0])
    ax1.view_init(30, -130)
    plt.colorbar(p, ax=ax1)
    ax1.set_title('Outputs')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')

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
        plot_sample(sid, source_file, show_title=False)
