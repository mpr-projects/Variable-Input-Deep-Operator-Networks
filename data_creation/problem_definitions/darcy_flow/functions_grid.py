import numpy as np
from functions_base import get_input, solve, finish


def setup(parent):
    settings = parent.settings
    
    # used for randomly drawing initial conditoins
    assert 'seed' in settings, "You must set a seed for creating initial conditions."
    parent.state['input_rng'] = np.random.default_rng(settings['seed'])

    # number of points in temporal domain
    n_x_points_sim = settings['n_x_points_sim']
    n_x_points_input = settings['n_x_points_input']
    n_x_points_output = settings['n_x_points_output']

    assert (n_x_points_sim - 1) % (n_x_points_input - 1) == 0, \
        "Invalid number of n_x_points_input."
    
    assert (n_x_points_sim - 1) % (n_x_points_output - 1) == 0, \
        "Invalid number of n_x_points_output."

    n_x_step_input = (n_x_points_sim - 1) // (n_x_points_input - 1)
    parent.state['n_x_step_input'] = n_x_step_input
    
    n_x_step_output = (n_x_points_sim - 1) // (n_x_points_output - 1)
    parent.state['n_x_step_output'] = n_x_step_output

    # set up grid, use same values for x and y
    x_values, dx = np.linspace(0, 1, num=n_x_points_sim, endpoint=True, retstep=True)

    # save coordinates in parent so we have access when
    # calculating the solution
    parent.state['x_coords'] = x_values
    parent.state['dx'] = dx

    # save coordinates in output file
    X, Y = np.meshgrid(x_values, x_values, indexing='ij')
    xy_values = np.stack((X, Y), axis=-1)
    
    parent.writer.add_data('input_coords', xy_values[::n_x_step_input, ::n_x_step_input], 'inputs')
    parent.writer.add_meta('n_input_coords', n_x_points_input**2)

    parent.writer.add_data('output_coords', xy_values[::n_x_step_output, ::n_x_step_output], 'outputs')
    parent.writer.add_meta('n_output_coords', n_x_points_output**2)


def scatter_sample_model(coords, model, cid):
    preds = model(coords)[:, cid].detach().cpu()
    p = plt.scatter(coords[:, 1].cpu(), coords[:, 0].cpu(), c=preds)
    plt.title(f'Component/Sample {cid}')
    plt.colorbar(p)
    plt.axis('equal')
    plt.show()


def process_input(parent):
    step = parent.state['n_x_step_input']
    return {f"input_{parent.sid}": parent.input[::step, ::step, None]}


def process_output(parent):
    step = parent.state['n_x_step_output']
    # visualize_result(parent.input[::step, ::step], parent.output[::step, ::step])
    return {f"output_{parent.sid}": parent.output[::step, ::step, None]}
