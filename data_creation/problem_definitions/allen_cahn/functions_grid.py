import numpy as np

from functions_base import get_grid, get_input, solve, finish


def setup(parent):
    """
    Create analytical solution for the travelling wave
    solution fo the Allen-Cahn problem.

    """
    settings = parent.settings
    
    # used for randomly drawing initial conditoins
    assert 'seed' in settings, "You must set a seed for creating initial conditions."
    parent.state['input_rng'] = np.random.default_rng(settings['seed'])

    # list with two entries, each having two components (lower and upper
    # bound of spatial domain)
    spatial_domain = settings['domain']

    # number of points in spatial and temporal domain
    n_x_points_sim = settings['n_x_points_sim']  # per dimension
    n_x_points_input = settings.get('n_x_points_input', n_x_points_sim)  # per dimension
    n_x_points_output = settings.get('n_x_points_output', n_x_points_sim)  # per dimension
    n_t_points = settings['n_t_points']

    assert (n_x_points_sim - 1) % (n_x_points_input - 1) == 0, \
        "Invalid number of n_x_points_input."
    
    assert (n_x_points_sim - 1) % (n_x_points_output - 1) == 0, \
        "Invalid number of n_x_points_output."

    n_x_step_input = (n_x_points_sim - 1) // (n_x_points_input - 1)
    parent.state['n_x_step_input'] = n_x_step_input
    
    n_x_step_output = (n_x_points_sim - 1) // (n_x_points_output - 1)
    parent.state['n_x_step_output'] = n_x_step_output

    # end time
    T = settings['T']

    # set up grid
    txy_vals = get_grid(
        [(0, T), spatial_domain[0], spatial_domain[1]],
        [n_t_points, n_x_points_sim, n_x_points_sim])

    # save coordinates in parent so we have access when
    # calculating the solution
    parent.state['coords'] = txy_vals

    # save coordinates in output file
    parent.writer.add_data('input_coords', txy_vals[0, ::n_x_step_input, ::n_x_step_input, 1:], 'inputs')
    parent.writer.add_meta('n_input_coords', n_x_points_input**2)

    parent.writer.add_data('output_coords', txy_vals[:, ::n_x_step_output, ::n_x_step_output], 'outputs')
    parent.writer.add_meta('n_output_coords', n_t_points * n_x_points_output**2)

    # will be used to save epsilon values
    parent.state['epsilon_values'] = list()


def process_input(parent):
    step = parent.state['n_x_step_input']
    return {f"input_{parent.sid}": parent.input[::step, ::step]}


def process_output(parent):
    step = parent.state['n_x_step_output']
    return {f"output_{parent.sid}": parent.output[:, ::step, ::step]}


