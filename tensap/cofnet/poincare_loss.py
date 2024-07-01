import numpy as np
import scipy


def poincare_loss(jac_u, jac_g):
    """
    Evaluate the Poincare based loss for a general feature map from samples.
    For more details see Bigoni et al. 2022.

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

    Returns
    -------
    out : float
        Poincare loss.
    """
    N = jac_u.shape[0]
    out = 0
    for ju, jg in zip(jac_u, jac_g):
        proj = scipy.linalg.pinv(jg) @ jg
        out += np.linalg.norm(ju.T - proj @ ju.T) ** 2 / N
    return out

def poincare_loss_augmented(jac_u, jac_g, alpha=1):
    """
    Evaluate the Poincare based loss with dimension augmentation, for a general feature map using only samples.
    For more details see Verdiere et al. 2023.

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
    N, d = jac_u.shape[0], jac_u.shape[-1]
    out = np.linalg.norm(jac_u)**2 / N
    for ju, jg in zip(jac_u, jac_g):
        jgju = jg @ ju.T
        gram = alpha * np.eye(d) + jg @ jg.T
        out -= np.linalg.norm(jgju.T @ np.linalg.solve(gram, jgju))**2 / N
    return out


class PoincareLoss:
    """
    Class PoincareLoss.
    Class implementing the Poincare based loss from Bigoni et al. 2022.

    Attributes
    ----------
    jac_u : numpy.ndarray, optional
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        Has shape (N, m, d).

    """

    def __init__(self, jac_u=None, jac_g=None):
        self.jac_u = jac_u
        self.jac_g = jac_g

    def eval(self, jac_u=None, jac_g=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_g is None: jac_g = self.jac_g
        return self.poincare_loss(jac_u, jac_g)