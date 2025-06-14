import numpy as np
from functions_base import get_input, delete_indices, solve

# random number of randomly chosen points

def setup(parent):
    settings = parent.settings
    
    # used for randomly drawing initial conditoins
    assert 'seed' in settings, "You must set a seed for creating initial conditions."
    parent.state['input_rng'] = np.random.default_rng(settings['seed'])

    # used to randomly draw grid perturbation
    coord_seed = settings.get('coord_seed', 987)
    parent.state['coord_rng'] = np.random.default_rng(coord_seed)

    # number of points in temporal domain
    n_x_points_sim = settings['n_x_points_sim']
    n_x_points_grid = settings['n_x_points_grid']  # per dimension
    n_x_points_delete_range = settings['n_x_points_delete_range']
    n_x_points_output = settings['n_x_points_output']  # number of points per dimension

    assert (n_x_points_sim - 1) % (n_x_points_grid - 1) == 0, \
        "Invalid number of n_x_points_grid."
    
    step = (n_x_points_sim - 1) // (n_x_points_grid - 1)
    parent.state['step'] = step

    if isinstance(n_x_points_delete_range, int):
        n_x_points_delete_range = [n_x_points_delete_range, n_x_points_delete_range]
        settings['n_x_points_delete_range'] = n_x_points_delete_range

    assert len(n_x_points_delete_range) == 2, (
        "You must provide a list with two entries, lower and upper bound for the number."
        " of x points to delete.")

    assert (n_x_points_sim - 1) % (n_x_points_output - 1) == 0, \
        "Invalid number of n_x_points_output."

    n_x_step_output = (n_x_points_sim - 1) // (n_x_points_output - 1)
    parent.state['n_x_step_output'] = n_x_step_output

    # set up grid, use same values for x and y
    x_values, dx = np.linspace(0, 1, num=n_x_points_sim, endpoint=True, retstep=True)

    # save simulation coordinates in parent so we have access when
    # calculating the solution
    parent.state['x_coords'] = x_values
    parent.state['dx'] = dx

    # save coordinates
    X, Y = np.meshgrid(x_values, x_values, indexing='ij')
    xy_values = np.stack((X, Y), axis=-1)

    parent.writer.add_data(
        'output_coords', xy_values[::n_x_step_output, ::n_x_step_output].reshape(-1, 2), 'outputs')

    parent.writer.add_data('output_coords_0', np.arange(n_x_points_output**2).reshape(-1, 1), 'outputs')

    xy_values = xy_values[::step, ::step].reshape(-1, 2)
    parent.writer.add_data('input_coords', xy_values, 'inputs')


def process_input(parent):
    settings = parent.settings
    n_x_points_grid = settings['n_x_points_grid']
    step = parent.state['step']

    coord_rng = parent.state['coord_rng']
    output_indices = delete_indices(parent, coord_rng)
    
    parent.writer.add_data(f'input_coords_{parent.sid}', output_indices.reshape(-1, 1), 'inputs')

    return {f"input_{parent.sid}": parent.input[::step, ::step].reshape(-1, 1)[output_indices]}


def process_output(parent):
    step = parent.state['n_x_step_output']
    return {f"output_{parent.sid}": parent.output[::step, ::step, None].reshape(-1, 1)}


def finish(parent):
    # add attribute 'coord_id' to inputs and outputs
    for sid in range(parent.settings['n_samples']):
        parent.writer.add_meta('coord_id', sid, f'inputs/input_{sid}')
        parent.writer.add_meta('coord_id', 0, f'outputs/output_{sid}')
