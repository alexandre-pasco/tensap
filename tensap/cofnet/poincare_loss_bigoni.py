import numpy as np
import scipy
from tensap.cofnet.poincare_loss import poincare_loss


def eval_jac_g(G, jac_basis):
    """
    Compute evaluations of jac_g from evaluation of jac_basis, where g = G.T @ basis.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_basis : numpy.ndarray
        N Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).

    Returns
    -------
    out : numpy.ndarray
        Has shape (N, m, d)
    """
    K = jac_basis.shape[1]
    if G.ndim == 1:
        Gmat = G.reshape((K, G.shape[0] // K), order='F')  # column-major ordering
    else:
        Gmat = G
    jac_g = np.einsum("ji, kjl", Gmat, jac_basis)
    jac_g = np.moveaxis(jac_g, 0, 1)
    return jac_g

def poincare_loss_bigoni(G, jac_u, jac_basis, jac_g=None):
    """
    Evaluate the Poincare based loss for the feature map G.T @ basis, as described in Bigoni et al. 2022.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        N Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    jac_basis : numpy.ndarray
        N Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    out : float
        Poincare based loss.
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    out = poincare_loss(jac_u, jac_g)
    return out

def poincare_loss_bigoni_gradient(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the gradient of the Poincaré-based of G.T @ basis with respect to G.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    gradient : numpy.ndarray
        The gradient of the loss.
        Has same shame as G.

    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    N, m, d = jac_g.shape
    K = jac_basis.shape[1]
    gradients = np.zeros((N, K, m))
    for k, jb, jg, ju in zip(np.arange(N), jac_basis, jac_g, jac_u):
        ug, sg, vgh = np.linalg.svd(jg)
        ug, vgh = ug[:, :m], vgh[:m, :]
        res1 = jb @ (np.eye(d) - vgh.T @ vgh) @ ju.T
        res2 = ug @ np.diag(1 / sg) @ vgh @ ju.T
        gradients[k] = - 2 * res1 @ res2.T
    gradient = gradients.mean(axis=0)
    gradient = gradient.reshape(G.shape, order="F")
    return gradient

def eval_SG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication Sigma(G)X as described in Bigoni et al. 2022, where
    X is a column-major vectorized matrix of same size as G.
    In other words, it computes Sigma(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Sx : numpy.ndarray
        Has same shape as X.
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    Xmat = X.reshape((K, m), order='F')  # column-major ordering
    Sx = np.zeros(Xmat.shape)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        Y = np.linalg.solve(GBG.T, Xmat.T)
        Sx += jb @ jb.T @ np.linalg.solve(GBG.T, GAG.T @ Y).T / N
    Sx = Sx.reshape(X.shape, order='F')  # column-major ordering
    return Sx

def eval_HG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication H(G)X as described in Bigoni et al. 2022, where
    X is a column-major vectorized matrix of same size as G.
    In other words, it computes H(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Hx : numpy.ndarray
        Has same shape as X.
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    Xmat = X.reshape((K, m), order='F')  # column-major ordering
    Hx = np.zeros(Xmat.shape)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jbju = jb @ ju.T
        GBG = jg @ jg.T
        Y = np.linalg.solve(GBG.T, Xmat.T)
        Hx += jbju @ jbju.T @ Y.T / N
    Hx = Hx.reshape(X.shape, order='F')  # column-major ordering
    return Hx

def eval_SG_diag(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the diagnoal of the matrix Sigma(G) from Bigoni et al. 2022.
    It is used for preconditioning the conjugate gradient used to apply the inverse of Sigma(G).

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    diag : numpy.ndarray
        Has shape (K*m, ).
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    diag = np.zeros(K*m)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        M = np.linalg.solve(GBG.T, GAG.T).T
        M = np.linalg.solve(GBG, M)
        diag1 = np.diag(M)
        diag2 = np.linalg.norm(jb, axis=1)
        diag += np.kron(diag1, diag2) / N
    return diag

def eval_SGinv_X(G, X, jac_u, jac_basis, jac_g=None, **kwargs):
    """
    Apply the inverse of the matrix Sigma(G) from Bigoni et al. 2022 to a vector X.
    The conjugate gradient method is used with Jacobi preconditioning.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).
    **kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg.

    Returns
    -------
    out : numpy.ndarray
        Has same shape as X.
    """
    N, K, d = jac_basis.shape
    Km = np.prod(G.shape)
    m = Km // K
    Xvec = X.reshape(Km, order='F')
    matvec = lambda Y: eval_SG_X(G, Y, jac_u, jac_basis, jac_g)
    sigma = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec)
    diag = eval_SG_diag(G, jac_u, jac_basis)
    M = scipy.sparse.diags(1 / diag)
    out, info = scipy.sparse.linalg.cg(sigma, Xvec, M=M, **kwargs)
    out = out.reshape(X.shape, order='F')
    return out

def eval_SG_HG_full(G, jac_u, jac_basis, jac_g=None):
    """
    Build the full matrices Sigma(G) and H(G) from Bigoni et al. 2022.
    This should only be used for debugging or testing purpose.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    S : numpy.ndarray
        Has shape (K*m, K*m).
    H : numpy.ndarray
        Has shape (K*m, K*m).
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    S = np.zeros((K*m, K*m))
    H = np.zeros((K*m, K*m))
    for ju, jb, jg in zip(jac_u, jac_basis, jac_g):
        jbju = jb @ ju.T
        A = jbju @ jbju.T
        B = jb @ jb.T
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        GBG_inv = np.linalg.inv(GBG)
        S += np.kron(GBG_inv @ GAG @ GBG_inv, B) / N
        H += np.kron(GBG_inv, A) / N
    return S, H

def eval_HessG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication Hess(G) @ X where
    X is a column-major vectorized matrix of same size as G.
    In other words, it computes Hess(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) of (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray.
        Has shape (N, n, d)
    jac_basis : numpy.ndarray.
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    hess : numpy.ndarray
        Has same shape as X.
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    if X.ndim == 1:
        Xmat = X.reshape((K, m), order='F')  # column-major ordering
    else:
        Xmat = X
    hess = 0 * Xmat
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        grad_u = ju.T
        djg = Xmat.T @ jb
        jg_inv = scipy.linalg.pinv(jg)
        p = np.eye(d) - jg_inv @ jg
        res1 = 2 * jg_inv.T @ (djg.T @ jg_inv.T - jg_inv @ djg @ p) @ grad_u @ grad_u.T @ p
        res2 = 2 * jg_inv.T @ grad_u @ grad_u.T @ (jg_inv @ djg @ p + p @ djg.T @ jg_inv.T)
        hess += jb @ (res1 + res2).T / N

    hess = hess.reshape(X.shape, order='F')  # column-major ordering
    return hess

def eval_HessG_diag(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the diagonal of the matrix Hess(G).
    It is used for preconditioning the conjugate gradient used to apply the inverse of Hess(G).

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    diag : numpy.ndarray
        Has shape (K*m, ).
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    diag_mat = np.zeros((K, m))
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        grad_u = ju.T
        jg_inv = scipy.linalg.pinv(jg)
        p = np.eye(jb.shape[1]) - jg_inv @ jg
        diag_mat += (jb @ p @ grad_u @ grad_u.T @ jg_inv) * (jb @ jg_inv)
        diag_mat -= (jb @ p @ grad_u) ** 2 @ np.ones((ju.shape[0], 1)) @ np.diag(jg_inv.T @ jg_inv).reshape(1, -1)
        diag_mat += (jb @ jg_inv) * (jb @ p @ grad_u @ grad_u.T @ jg_inv)
        diag_mat += (jb @ p) ** 2 @ np.ones((d, 1)) @ np.diag(jg_inv.T @ grad_u @ grad_u.T @ jg_inv).reshape(1, -1)
    diag_mat = 2 * diag_mat / N
    diag = diag_mat.flatten(order="F")
    return diag

def eval_HessG_full(G, jac_u, jac_basis, jac_g=None):
    """
    Build the full matrices Hess(G).
    This should only be used for debugging or testing purpose.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K, m).
    jac_u : numpy.ndarray
        Has shape (N, n, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Hess : numpy.ndarray
        Has shape (K*m, K*m).
    """
    if jac_g is None:
        jac_g = eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    hess = np.zeros((K*m, K*m))
    for i in range(K*m):
        X = np.zeros(K*m)
        X[i] = 1
        hess[i,:] = eval_HessG_X(G, X, jac_u, jac_basis, jac_g)
    return hess

class PoincareLossBigoni:
    """
    Class PoincareLossBigoni.
    Class implementing the Poincare based loss from Bigoni et al. 2022 for the specific case where
    g = G.T @ basis.

    Attributes
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    basis : instance of class with eval method
        Object such that g(x) = G.T @ basis.eval(x)
    """

    def __init__(self, jac_u, jac_basis, basis=None):
        self.jac_u = jac_u
        self.jac_basis = jac_basis
        self.basis = basis

    def eval(self, G, jac_g=None):
        return poincare_loss_bigoni(G, self.jac_u, self.jac_basis, jac_g)

    def eval_gradient(self, G, jac_g=None):
        return poincare_loss_bigoni_gradient(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_X(self, G, X, jac_g=None):
        return eval_SG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_HG_X(self, G, X, jac_g=None):
        return eval_HG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_diag(self, G, jac_g=None):
        return eval_SG_diag(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_HG_full(self, G, jac_g=None):
        return eval_SG_HG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_X(self, G, X, jac_g=None):
        return eval_HessG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_diag(self, G, jac_g=None):
        return eval_HessG_diag(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_full(self, G, jac_g=None):
        return eval_HessG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SGinv_X(self, G, X, jac_g=None, **kwargs):
        return eval_SGinv_X(G, X, self.jac_u, self.jac_basis, jac_g, **kwargs)
