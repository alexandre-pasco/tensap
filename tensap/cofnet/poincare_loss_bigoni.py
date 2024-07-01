import numpy as np
import scipy
from poincare_loss import poincare_loss


def compute_jac_g(G, jac_basis):
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
    jac_g = np.einsum("ij, kjl", G.T, jac_basis)
    jac_g = np.moveaxis(jac_g, 0, 1)
    return jac_g

def poincare_loss_bigoni(G, jac_u, jac_basis):
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

    Returns
    -------
    out : float
        Poincare based loss.
    """
    jac_g = compute_jac_g(G, jac_basis)
    out = poincare_loss(jac_u, jac_g)
    return out

def poincare_loss_bigoni_augmented(G, jac_u, jac_basis, alpha=1):
    """
    Evaluate the Poincare based loss for the feature map G.T @ basis, as described in Bigoni et al. 2022,
    with dimension augmentation as described in Verdiere et al. 2023.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_g : numpy.ndarray
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        Has shape (N, m, d).
    alpha : float
        The regularisation parameter
    Returns
    -------
    out : float
        Poincare loss.
    """
    jac_g = compute_jac_g(G, jac_basis)
    out = poincare_loss_bigoni_augmented(jac_u, jac_g, alpha)
    return out


class PoincareLossBigoni:
    """
    Class PoincareLossBigoni.
    Class implementing the Poincare based loss from Bigoni et al. 2022 for the specific case where
    g = G.T @ phi.

    Attributes
    ----------
    jac_u : numpy.ndarray, optional
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray, optional
        Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).

    """

    def __init__(self, jac_u=None, jac_basis=None, basis=None):
        self.basis = basis
        self.jac_u = jac_u
        self.jac_basis = jac_basis


    def eval(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return poincare_loss_bigoni(G, jac_u, jac_basis)