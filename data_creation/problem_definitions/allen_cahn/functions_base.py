import math
import torch
import numpy as np


def get_grid(domain, n_points):
    """
    Set up grid points from domain information.

    """
    grid = list()

    for n, d in zip(n_points, domain):
        grid.append(np.linspace(d[0], d[1], n))

    grid = np.meshgrid(*grid, indexing='ij')
    return np.stack(grid, axis=-1)


def delete_indices(parent, coord_rng=None):
    settings = parent.settings
    n_x_points_sim = settings['n_x_points_sim']  # per dimension
    n_x_points_save = settings['n_x_points_grid']  # per dimension

    if coord_rng is None:
        coord_seed = settings.get('coord_seed', 987)
        coord_rng = np.random.default_rng(coord_seed)

    delete_range = settings['n_x_points_delete_range']
    n_del = coord_rng.integers(delete_range[0], delete_range[1], endpoint=True)

    idx_vals = coord_rng.permutation(n_x_points_save**2)

    if n_del == 0:
        return idx_vals

    return idx_vals[:-n_del]


def get_perturbed_indices(parent, coord_rng=None):
    settings = parent.settings
    n_x_points_sim = settings['n_x_points_sim']  # per dimension
    n_x_points_save = settings['n_x_points_grid']  # per dimension

    step = (n_x_points_sim - 1) // (n_x_points_save - 1)

    if coord_rng is None:
        coord_seed = settings.get('coord_seed', 987)
        coord_rng = np.random.default_rng(coord_seed)

    max_pert = settings['max_perturbation']
    x_pert = coord_rng.integers(-max_pert, max_pert, size=n_x_points_save**2, endpoint=True)
    y_pert = coord_rng.integers(-max_pert, max_pert, size=n_x_points_save**2, endpoint=True)

    idx_vals = np.arange(n_x_points_save)
    i_vals, j_vals = np.meshgrid(idx_vals, idx_vals, indexing='ij')
    i_vals, j_vals = i_vals.flatten(), j_vals.flatten()

    i_vals = i_vals * n_x_points_sim * step + y_pert * n_x_points_sim
    j_vals = j_vals * step + x_pert

    i_vals = np.minimum(np.maximum(i_vals, 0), n_x_points_sim**2 - n_x_points_sim)
    j_vals = np.minimum(np.maximum(j_vals, 0), n_x_points_sim - 1)

    return i_vals + j_vals


def process_perturbed_indices(indices, n_grid_points, n_points_to_save, coord_rng):
    indices = np.unique(indices)

    if len(indices) < n_points_to_save:
        n_required = n_points_to_save - len(indices)
        unused_indices = [idx for idx in range(n_grid_points) if idx not in indices]
        unused_indices = coord_rng.permutation(unused_indices)[:n_required]
        indices = np.concatenate((indices, unused_indices))

    coord_rng.shuffle(indices)
    return indices[:n_points_to_save]


def get_input(parent):
    # list of two entries, lower and upper bound of epsilon-domain
    rng = parent.state['input_rng']
    eps_domain = parent.settings['eps_domain']
    eps = rng.uniform(eps_domain[0], eps_domain[1])
    parent.state['epsilon_values'].append(eps)

    # velocity of wave depends on epsilon
    s = 3 / np.sqrt(2) / eps

    # shift initial condition across spatial domain
    domain = parent.settings['domain']
    ic_offset_x = rng.uniform(domain[0][0], domain[0][1])
    ic_offset_y = rng.uniform(domain[1][0], domain[1][1])

    # rotate wave (sum of squared weights must equal 1)
    ic_rotation_x = rng.uniform()
    ic_rotation_y = (1 - ic_rotation_x**2)**0.5

    ic_rot_sign_x = 1 if rng.uniform() > 0.5 else -1
    ic_rot_sign_y = 1 if rng.uniform() > 0.5 else -1

    ic_rotation_x *= ic_rot_sign_x
    ic_rotation_y *= ic_rot_sign_y

    # compute solution
    t_coords = parent.state['coords'][..., 0]
    x_coords = parent.state['coords'][..., 1]
    y_coords = parent.state['coords'][..., 2]

    shifted_x = x_coords - ic_offset_x
    shifted_y = y_coords - ic_offset_y
    res = (ic_rotation_x * shifted_x + ic_rotation_y * shifted_y - s * t_coords) / 2 / np.sqrt(2) / eps
    res = 0.5 - 0.5 * np.tanh(res[..., None])

    # save entire solution in parent
    parent.state['current_solution'] = res

    # return initial condition
    return res[0]


def solve(parent):
    # already solved in 'get_inputs'
    print('finished', parent.sid, end='\r')
    return parent.state['current_solution']


def finish(parent):
    parent.save_data(
        {'epsilon_values': np.array(parent.state['epsilon_values'])}, 'inputs')
