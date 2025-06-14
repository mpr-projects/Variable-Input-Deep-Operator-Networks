import numpy as np

import functions_base
from functions_base import get_grid, delete_indices, get_input, solve


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
    n_x_points_grid = settings['n_x_points_grid']  # per dimension
    n_x_points_delete_range = settings['n_x_points_delete_range']  # in total
    n_x_points_output = settings['n_x_points_output']  # per dimension
    n_t_points = settings['n_t_points']

    assert (n_x_points_sim - 1) % (n_x_points_grid - 1) == 0, \
        "Invalid number of n_x_points_grid."
    
    step_input = (n_x_points_sim - 1) // (n_x_points_grid - 1)
    parent.state['step_input'] = step_input

    if isinstance(n_x_points_delete_range, int):
        n_x_points_delete_range = [n_x_points_delete_range, n_x_points_delete_range]

    assert len(n_x_points_delete_range) == 2, (
        "You must provide a list with two entries, lower and upper bound for the number."
        " of x points to save.")

    assert (n_x_points_sim - 1) % (n_x_points_output - 1) == 0, \
        "Invalid number of n_x_points_output."

    step = (n_x_points_sim - 1) // (n_x_points_output - 1)
    parent.state['n_x_step_output'] = step

    # end time
    T = settings['T']

    # set up grid
    txy_vals = get_grid(
        [(0, T), spatial_domain[0], spatial_domain[1]],
        [n_t_points, n_x_points_sim, n_x_points_sim])

    # save coordinates in parent so we have access when
    # calculating the solution
    parent.state['coords'] = txy_vals

    parent.writer.add_data('input_coords', txy_vals[0, ::step_input, ::step_input, 1:].reshape(-1, 2), 'inputs')

    parent.writer.add_data('output_coords', txy_vals[:, ::step, ::step].reshape(-1, 3), 'outputs')
    parent.writer.add_data(
        'output_coords_0', np.arange(n_t_points * n_x_points_output**2)[:, None], 'outputs')

    parent.writer.add_meta('n_output_coords', n_t_points * n_x_points_output**2)

    # will be used to save epsilon values
    parent.state['epsilon_values'] = list()


def process_input(parent):
    settings = parent.settings
    n_x_points_grid = settings['n_x_points_grid']
    n_t_points = settings['n_t_points']

    step = parent.state['step_input']

    coord_rng = parent.state['coord_rng']
    output_indices = delete_indices(parent, coord_rng)

    parent.writer.add_data(f'input_coords_{parent.sid}', output_indices.reshape(-1, 1), 'inputs')

    return {f"input_{parent.sid}": parent.input[::step, ::step].reshape(-1, 1)[output_indices]}


def process_output(parent):
    step = parent.state['n_x_step_output']
    return {f"output_{parent.sid}": parent.output[:, ::step, ::step].reshape(-1, 1)}


def finish(parent):
    # save epsilon values
    functions_base.finish(parent)

    # add attribute 'coord_id' to inputs and outputs
    for sid in range(parent.settings['n_samples']):
        parent.writer.add_meta('coord_id', sid, f'inputs/input_{sid}')
        parent.writer.add_meta('coord_id', 0, f'outputs/output_{sid}')
