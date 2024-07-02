import numpy as np
import scipy
from tensap.cofnet.poincare_loss_bigoni import *


def iteration_qn_bigoni(G, jac_u, jac_basis, **kwargs):
    N, K, d = jac_basis.shape
    Gmat = G.reshape(K, -1, order='F')
    jac_g = eval_jac_g(Gmat, jac_basis)
    b = eval_HG_X(Gmat, Gmat, jac_u, jac_basis, jac_g)
    Gaux = eval_SGinv_X(Gmat, b, jac_u, jac_basis, jac_g, **kwargs)
    Gnext = np.linalg.svd(Gaux, full_matrices=False)[0]  # orthonormalize
    # M = Gaux.T @ R @ Gaux  # orthonormalization
    # Mcholinv = scipy.linalg.inv(scipy.linalg.cholesky(M))
    # Gnext = Gaux @ Mcholinv
    return Gnext

def minimize_qn_bigoni(G0, jac_u, jac_basis, maxiter_qn=100, tol_qn=1e-5, verbosity=2, **kwargs):
    N, K, d = jac_basis.shape
    G0mat = G0.reshape(K, -1, order='F')
    Gnow = G0mat
    i = -1
    delta = np.inf
    if verbosity >= 1:
        print("Optimizing Poincare loss with QN from Bigoni et al.")
    while i < maxiter_qn and delta >= tol_qn:
        i = i+1
        Gnext = iteration_qn_bigoni(Gnow, jac_u, jac_basis, **kwargs)
        delta = np.linalg.norm(Gnext - Gnow)
        # delta = 1 - np.linalg.svd(Gnext.T @ Gnow)[1].min()
        Gnow = Gnext
        if verbosity >= 2:
            err = poincare_loss_bigoni(Gnow, jac_u, jac_basis)
            print(f"| Iter:{i} loss:{err:.3e} step_size:{delta:.3e}")
    return Gnow

def minimize_scipy(G0, jac_u, jac_basis, verbosity=1, **kwargs):
    N, K, d = jac_basis.shape
    G0vec = G0.flatten(order='F')  # column-major ordering

    # functions used in scipy.optimize.minimize
    def loss(x):
        return poincare_loss_bigoni(x, jac_u, jac_basis)
    def loss_grad(x):
        return poincare_loss_bigoni_gradient(x, jac_u, jac_basis)
    def hessp(x, p):
        return eval_SG_X(x, p, jac_u, jac_basis)
    def orthonormalize(x):
        xmat = np.linalg.svd(x.reshape(G0.shape), full_matrices=False)[0]
        x[:] = xmat.flatten("F")

    def print_info(x):
        l = loss(x)
        dl = np.linalg.norm(loss_grad(x))
        print(f"| loss_train={l:.3e} | norm_grad_loss_train={dl:.3e}")

    def callback(x):
        orthonormalize(x)
        if verbosity >= 1:
            print_info(x)

    res = scipy.optimize.minimize(fun=loss, x0=G0vec, jac=loss_grad, hessp=hessp, callback=callback, **kwargs)
    Gfit = res.x.reshape(G0.shape)
    return Gfit

def minimize_pymanopt(G0, jac_u, jac_basis, use_precond=True, pmo_kwargs={}, precond_kwargs={}):
    K = jac_basis.shape[1]
    if G0.ndim == 1:
        G0mat = G0.Reshape((K, G0.shape[0] // K))
    else:
        G0mat = G0
    m = G0mat.shape[1]
    problem, optimizer = build_pymanopt_problem(jac_u, jac_basis, m, use_precond, pmo_kwargs, precond_kwargs)
    optim_result = optimizer.run(problem, initial_point=G0)
    result = optim_result.point
    return result

def build_pymanopt_problem(jac_u, jac_basis, m, use_precond=True, pmo_kwargs={}, precond_kwargs={}):
    import pymanopt as pmo
    K = jac_basis.shape[1]
    manifold = pmo.manifolds.grassmann.Grassmann(K, m)
    @pmo.function.numpy(manifold)
    def cost(G): return poincare_loss_bigoni(G, jac_u, jac_basis)

    @pmo.function.numpy(manifold)
    def euclidean_gradient(G): return poincare_loss_bigoni_gradient(G, jac_u, jac_basis)

    precond = None
    if use_precond:
        def precond(G, x): return eval_SGinv_X(G, x, jac_u, jac_basis, **precond_kwargs)

    problem = pmo.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, preconditioner=precond)
    # line_search = pmo.optimizers.line_search.AdaptiveLineSearcher(max_iterations=50) # can be provided to optimizer
    optimizer = pmo.optimizers.conjugate_gradient.ConjugateGradient(**pmo_kwargs, line_searcher=None)

    return problem, optimizer


def eval_geig_matrices(G, jac_u, jac_basis, weight=0):
    """
    Build the matrices for my generalized eigh problem.

    Parameters
    ----------
    jac_u
    jac_basis
    G

    Returns
    -------

    """
    K, d = jac_basis.shape[1:]
    A = np.zeros((K, K))
    B = np.zeros((K, K))
    for jb, ju in zip(jac_basis, jac_u):
        if G is None:
            P_G = np.zeros((d, d))
        else:
            jg = G.T @ jb
            P_G = jg.T @ scipy.linalg.pinv(jg.T)
        P_G_perp = np.eye(d) - P_G
        v1 = jb @ P_G_perp @ ju.T / np.linalg.norm(P_G_perp @ ju.T)
        v2 = jb @ P_G
        Ax = v1 @ v1.T + v2 @ v2.T
        Bx = jb @ jb.T
        if weight == 0:
            c = 1.
        elif weight == 1:
            c = np.linalg.norm(P_G_perp @ ju.T)**2
        elif weight == 2:
            c = np.linalg.norm(ju.T)**2
        else:
            raise NotImplementedError()
        A += c * Ax / jac_u.shape[0]
        B += c * Bx / jac_u.shape[0]
    return A, B
