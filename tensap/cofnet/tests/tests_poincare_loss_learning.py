
import numpy as np
import tensap
from tensap.cofnet.poincare_loss_learning import *


def _build_test_case(d=5, n=2, N=50, m=2):
    # Build a multivariate polynomial basis with total degree at most 2
    x = np.random.RandomState(0).uniform(-1, 1, size=(N, d))
    h = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(3))
    H = tensap.FunctionalBases.duplicate(h, d)
    I0 = tensap.MultiIndices.with_bounded_norm(d, 1, 2).remove_indices(0)
    basis = tensap.SparseTensorProductFunctionalBasis(H, I0)
    K = basis.eval(x).shape[1]

    G0 = np.linalg.svd(np.array(np.random.RandomState(0).normal(scale=1 / np.sqrt(K), size=(K, m))), full_matrices=False)[0]
    G1 = np.array(np.random.RandomState(1).normal(scale=1 / np.sqrt(K), size=(K, m)))
    G2 = np.array(np.random.RandomState(2).normal(scale=1 / np.sqrt(K), size=(K, m)))

    def g(x):
        return basis.eval(x) @ G0

    def jac_g(x):
        jac_basis = basis.eval_jacobian(x)
        out = np.array([G0.T @ jb for jb in jac_basis])
        return out

    def f(z):
        out = np.block([[np.sin(z.prod(axis=1))], [np.cos(z).prod(axis=1)]]).T
        return out

    def jac_f(z):
        out = np.zeros((z.shape[0], n, z.shape[1]))
        out[:,0, :] = np.array([np.cos(zi.prod()) * zi.prod() * np.ones(zi.shape[0]) / zi for zi in z])
        out[:,1, :] = np.array([- np.cos(zi).prod() * np.ones(zi.shape[0]) * np.sin(zi) / np.cos(zi) for zi in z])
        return out

    def u(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return f(g(x))

    def jac_u(x):
        if x.ndim == 1:
            x = x.reshape(-1)
        jac_fz = jac_f(g(x))
        jac_gx = jac_g(x)
        out = np.array([jf @ jg for jf, jg in zip(jac_fz, jac_gx)])
        return out

    loss = tensap.PoincareLossBigoni(jac_u(x), basis.eval_jacobian(x), basis)
    loss1 = tensap.PoincareLossBigoni(jac_u(x)[:, [0], :], basis.eval_jacobian(x), basis)
    loss2 = tensap.PoincareLossBigoni(jac_u(x)[:, [1], :], basis.eval_jacobian(x), basis)
    resolution = np.sqrt(np.finfo(G0.dtype).resolution)

    return G0, G1, G2, loss, loss1, loss2, resolution

def test_minimize_qn_bigoni():
    G_true, G1, _, loss, _, _, resolution = _build_test_case(m=1)
    G_fit = minimize_qn_bigoni(G1, loss.jac_u, loss.jac_basis, maxiter_qn=100, tol_qn=1e-7, verbosity=0)
    sigma = np.linalg.svd(G_fit.T @ G_true)[1].min()
    np.testing.assert_allclose(sigma, 1.)

def test_minimize_scipy():
    G_true, G1, _, loss, _, _, resolution = _build_test_case(m=1)
    G_fit = minimize_scipy(G1, loss.jac_u, loss.jac_basis, method="Newton-CG", verbosity=0)
    sigma = np.linalg.svd(G_fit.T @ G_true)[1].min()
    np.testing.assert_allclose(sigma, 1.)

def test_minimize_pymanopt():
    G_true, G1, _, loss, _, _, resolution = _build_test_case(m=1)
    G_fit = minimize_pymanopt(G1, loss.jac_u, loss.jac_basis, use_precond=True, pmo_kwargs={"max_iterations":50, "verbosity":0})
    sigma = np.linalg.svd(G_fit.T @ G_true)[1].min()
    np.testing.assert_allclose(sigma, 1.)

def test_geig():
    G_true, G1, _, loss, _, _, resolution = _build_test_case(m=1)
    A, B = eval_geig_matrices(None, loss.jac_u, loss.jac_basis)
    w, v = scipy.linalg.eigh(A, B)
    G_fit = np.linalg.svd(v[:,-1:], full_matrices=False)[0]
    sigma = np.linalg.svd(G_fit.T @ G_true)[1].min()
    np.testing.assert_allclose(sigma, 1.)

def test_geig_bis():
    G_true, G1, _, loss, _, _, resolution = _build_test_case(m=2)
    A, B = eval_geig_matrices(G_true[:,:1], loss.jac_u, loss.jac_basis)
    w, v = scipy.linalg.eigh(A, B)
    G_fit = np.linalg.svd(v[:,-2:], full_matrices=False)[0]
    sigma = np.linalg.svd(G_fit.T @ G_true)[1].min()
    np.testing.assert_allclose(sigma, 1.)


if __name__ == "__main__":
    test_minimize_qn_bigoni()
    test_minimize_scipy()
    test_minimize_pymanopt()
    test_geig()
    test_geig_bis()
