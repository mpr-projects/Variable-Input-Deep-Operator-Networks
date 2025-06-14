import numpy as np
from functions_base import get_input, solve, finish


def setup(parent):
    settings = parent.settings
    
    # used for randomly drawing initial conditoins
    assert 'seed' in settings, "You must set a seed for creating initial conditions."
    parent.state['input_rng'] = np.random.default_rng(settings['seed'])

    # need to ensure different datasets use same grid
    coord_seed = settings.get('coord_seed', 987)
    coord_rng = np.random.default_rng(coord_seed)

    # number of points in temporal domain
    n_x_points_sim = settings['n_x_points_sim']
    n_x_points_save = settings['n_x_points_input_save']  # total number of points
    n_x_points_output = settings['n_x_points_output']  # number of points per dimension

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

    # randomly select points to use for output
    X, Y = np.meshgrid(x_values, x_values, indexing='ij')
    xy_values = np.stack((X, Y), axis=-1)

    parent.writer.add_data('output_coords', xy_values[::n_x_step_output, ::n_x_step_output], 'outputs')
    parent.writer.add_meta('n_output_coords', n_x_points_output**2)

    output_indices = coord_rng.permutation(n_x_points_sim**2)[:n_x_points_save]
    parent.state['output_indices'] = output_indices
    xy_values = xy_values.reshape((n_x_points_sim**2, 2))[output_indices]
    
    parent.writer.add_data('input_coords', xy_values, 'inputs')
    parent.writer.add_meta('n_input_coords', n_x_points_save)


def process_input(parent):
    output_indices = parent.state['output_indices']
    return {f"input_{parent.sid}": parent.input.reshape(-1, 1)[output_indices]}


def process_output(parent):
    step = parent.state['n_x_step_output']
    return {f"output_{parent.sid}": parent.output[::step, ::step, None]}
