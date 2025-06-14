import numpy as np

import functions_base
from functions_base import get_grid, get_input, solve


def setup(parent):
    """
    Create analytical solution for the travelling wave
    solution fo the Allen-Cahn problem.

    """
    settings = parent.settings
    
    # used for randomly drawing initial conditoins
    assert 'seed' in settings, "You must set a seed for creating initial conditions."
    parent.state['input_rng'] = np.random.default_rng(settings['seed'])

    # used to randomly draw grid perturbation
    coord_seed = settings.get('coord_seed', 987)
    parent.state['coord_rng'] = np.random.default_rng(coord_seed)

    # list with two entries, each having two components (lower and upper
    # bound of spatial domain)
    spatial_domain = settings['domain']

    # number of points in spatial and temporal domain
    n_x_points_sim = settings['n_x_points_sim']  # per dimension
    n_x_points_save_range = settings['n_x_points_save_range']  # in total
    n_t_points = settings['n_t_points']

    if isinstance(n_x_points_save_range, int):
        settings['n_x_points_save_range'] = [n_x_points_save_range, n_x_points_save_range]
        n_x_points_save_range = settings['n_x_points_save_range']  # in total

    assert len(n_x_points_save_range) == 2, (
        "You must provide a list with two entries, lower and upper bound for the number."
        " of x points to save.")

    # end time
    T = settings['T']

    # set up grid
    txy_vals = get_grid(
        [(0, T), spatial_domain[0], spatial_domain[1]],
        [n_t_points, n_x_points_sim, n_x_points_sim])

    # save coordinates in parent so we have access when
    # calculating the solution
    parent.state['coords'] = txy_vals

    parent.writer.add_data('input_coords', txy_vals[0, ..., 1:].reshape(-1, 2), 'inputs')
    parent.writer.add_data('output_coords', txy_vals.reshape(-1, 3), 'outputs')

    # will be used to save epsilon values
    parent.state['epsilon_values'] = list()


def process_input(parent):
    settings = parent.settings
    n_x_points_sim = settings['n_x_points_sim']
    n_t_points = settings['n_t_points']

    n_range = settings['n_x_points_save_range']
    coord_rng = parent.state['coord_rng']

    n_points_to_save = coord_rng.integers(n_range[0], n_range[1], endpoint=True)

    output_indices = coord_rng.permutation(n_x_points_sim**2)[:n_points_to_save]
    parent.writer.add_data(f'input_coords_{parent.sid}', output_indices.reshape(-1, 1), 'inputs')

    output_indices_out = list()

    for tid in range(n_t_points):
        output_indices_out += list(tid * n_x_points_sim**2 + output_indices)

    output_indices_out = np.array(output_indices_out)
    parent.state['output_indices'] = output_indices_out
    parent.writer.add_data(f'output_coords_{parent.sid}', output_indices_out.reshape(-1, 1), 'outputs')

    return {f"input_{parent.sid}": parent.input.reshape(-1, 1)[output_indices]}


def process_output(parent):
    output_indices = parent.state['output_indices']
    return {f"output_{parent.sid}": parent.output.reshape(-1, 1)[output_indices]}


def finish(parent):
    # save epsilon values
    functions_base.finish(parent)

    # add attribute 'coord_id' to inputs and outputs
    for sid in range(parent.settings['n_samples']):
        parent.writer.add_meta('coord_id', sid, f'inputs/input_{sid}')
        parent.writer.add_meta('coord_id', sid, f'outputs/output_{sid}')
