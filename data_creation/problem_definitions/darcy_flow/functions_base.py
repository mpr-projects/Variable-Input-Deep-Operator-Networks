import math
import numpy as np

from scipy.fftpack import idct
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


"""
# Calculate the numerical solutions to the problem
# 
# $$ - \nabla \cdot \Big(a(x) \nabla u(x) \Big) = f(x) $$
# 
# for $x \in (0,1)^2$ with zero Dirichlet boundary condition
# 
# $$ u(x) = 0 $$
# 
# on $x \in \partial (0,1)^2$ as described in Li (2020) on page 8.
# 
"""


def scatter_sample_model(coords, model, cid):
    preds = model(coords)[:, cid].detach().cpu()
    p = plt.scatter(coords[:, 1].cpu(), coords[:, 0].cpu(), c=preds)
    plt.title(f'Component/Sample {cid}')
    plt.colorbar(p)
    plt.axis('equal')
    plt.show()


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
    """Draws a random field in the KL expansion as done in Li (2020)."""
    # code adapted directly from
    # - https://github.com/zongyi-li/fourier_neural_operator/blob/master/data_generation/darcy/GRF.m
    # - https://github.com/zongyi-li/fourier_neural_operator/blob/master/data_generation/darcy/demo.m

    x_values = y_values = parent.state['x_coords']
    rng = parent.state['input_rng']

    alpha, tau = 2, 3
    s = len(x_values)

    #% Random variables in KL expansion
    #xi = normrnd(0,1,s);
    xi = rng.standard_normal((s, s))

    #% Define the (square root of) eigenvalues of the covariance operator
    #[K1,K2] = meshgrid(0:s-1,0:s-1);
    K2, K1 = np.mgrid[0:s, 0:s]
    #coef = tau^(alpha-1).*(pi^2*(K1.^2+K2.^2) + tau^2).^(-alpha/2);
    coef = tau**(alpha-1) * (np.pi**2*(K1**2+K2**2) + tau**2)**(-alpha/2);  # *np.eye(s) not used in tau

    #% Construct the KL coefficients
    #L = s*coef.*xi;
    #L(1,1) = 0;
    L = s*coef*xi
    L[0, 0] = 0

    #U = idct2(L);
    norm_a = idct(idct(L.T, norm='ortho').T, norm='ortho')

    if parent.settings['use_thresholding']:
        norm_a[norm_a >= 0] = 12
        norm_a[norm_a < 0] = 3
    else:
        norm_a -= np.amin(norm_a)
        norm_a *= 10
        norm_a += 0.5

    return norm_a


def get_f(parent):
    n_x_points_sim = parent.settings['n_x_points_sim']
    return np.ones((n_x_points_sim, n_x_points_sim))


def first_derivative(parent, matrix, dim):
    """Expects a (2-dimensional) matrix and returns the first derivative using central differences (2nd order)."""
    dx = dy = parent.state['dx']
    ds = [dx, dy]
    mat = np.moveaxis(matrix, dim, -1)
    der = np.zeros_like(matrix)
    der[:, 1:-1] = (mat[:, 2:] - mat[:, :-2]) / 2 / ds[dim]
    return np.moveaxis(der, -1, dim)


def mat_to_vec(mat):
    return np.reshape(mat, -1)


def vec_to_mat(vec):
    n_x_points_sim = int(vec.size**0.5)
    return np.reshape(vec, (n_x_points_sim, n_x_points_sim))


def idx(parent, i, j):
    n_y_points_sim = parent.settings['n_x_points_sim']
    return i*n_y_points_sim + j


def build_matrix(parent, a1d, ax1d, ay1d):
    n_x_points_sim = n_y_points_sim = parent.settings['n_x_points_sim']
    dx = dy = parent.state['dx']
    
    row, col, val = list(), list(), list()

    aip1j = ax1d/2/dx + a1d/dx**2
    aim1j = a1d/dx**2 - ax1d/2/dx
    aijp1 = ay1d/2/dy + a1d/dy**2
    aijm1 = a1d/dy**2 - ay1d/2/dy
    aij = - 2*a1d/dx**2 - 2*a1d/dy**2

    for i in range(n_x_points_sim):
        for j in range(n_y_points_sim):
            cur_idx = idx(parent, i, j)

            if i < n_x_points_sim-1:
                row.append(cur_idx)
                col.append(idx(parent, i+1, j))
                val.append(aip1j[cur_idx])

            if i > 0:
                row.append(cur_idx)
                col.append(idx(parent, i-1, j))
                val.append(aim1j[cur_idx])

            if j < n_y_points_sim-1:
                row.append(cur_idx)
                col.append(idx(parent, i, j+1))
                val.append(aijp1[cur_idx])

            if j > 0:
                row.append(cur_idx)
                col.append(idx(parent, i, j-1))
                val.append(aijm1[cur_idx])

            row.append(cur_idx)
            col.append(cur_idx)
            val.append(aij[cur_idx])

    return coo_matrix((val, (row, col))).tocsr()


def _solve(parent, a2d, f2d):
    
    # prepare data
    a1d = mat_to_vec(a2d)
    
    ax2d = first_derivative(parent, a2d, 0)
    ax1d = mat_to_vec(ax2d)

    ay2d = first_derivative(parent, a2d, 1)
    ay1d = mat_to_vec(ay2d)
    
    # get matrix and solve system
    f1d = mat_to_vec(f2d)
    mat = build_matrix(parent, a1d, ax1d, ay1d)
    u1d = spsolve(mat, -f1d)
    u2d = vec_to_mat(u1d)
    
    return u2d


def visualize_result(a2d, u2d, redo=False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mpl.rcParams["figure.figsize"] = (10, 10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    min_val = min(np.amin(a2d), np.amin(u2d))
    max_val = max(np.amax(a2d), np.amax(u2d))

    im1 = ax1.imshow(a2d)  #, vmin=min_val, vmax=max_val)
    im2 = ax2.imshow(u2d)  #, vmin=min_val, vmax=max_val)
    #plt.colorbar(im1, ax=[ax1, ax2], orientation='horizontal')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    if redo:
        plt.title('not good enough')
    plt.show()
    
    
def solve(parent):
    u2d = _solve(parent, parent.input, get_f(parent))
        
    redo = False

    # redraw this sample if equation could not be solved
    if np.isnan(u2d).any():
        parent.valid_input = False
        redo = True

    if np.amin(u2d) < 0:
        parent.valid_input = False
        redo = True
            
    # visualize_result(parent.input, u2d, redo)
    # raise RuntimeError('..')
    return u2d


def finish(parent):
    return
