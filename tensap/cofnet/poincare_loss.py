import numpy as np
import scipy


class PoincareLossFunction:
    """
    Class PoincareLossFunction.
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

    @staticmethod
    def _eval(jac_u, jac_g):
        """
        Evaluate the Poincare based loss for a general feature map using only samples.
        
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
        loss : float
            Poincare based loss.
        """
        jac_ut = np.moveaxis(jac_u, -1, -2)
        proj = np.array([jg.T @ np.linalg.solve(jg @ jg.T, jg) for jg in jac_g])
        residual = jac_ut - np.einsum("kil,klj->kij", proj, jac_ut, optimize=True)
        losses = np.linalg.norm(residual, axis=(1,2)) ** 2
        loss = losses.mean()
        return loss

    def eval(self, jac_u=None, jac_g=None):
        """
        Evaluate the Poincare based loss for a general feature map using only samples.
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        jac_u : numpy.ndarray, optional
            Samples of the jacobian of the function to approximate.
            jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
            If None, self.jac_u is used instead.
            Has shape (N, n, d).
        jac_g : numpy.ndarray, optional
            Samples of the jacobian of the feature map, whose quality is assessed by the Poincare based loss.
            jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
            If None, self.jac_g is used instead.
            Has shape (N, m, d).

        Returns
        -------
        loss : float
            Poincare based loss.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_g is None: jac_g = self.jac_g
        return self._eval(self.jac_u, self.jac_g)


class RayleighPoincareLossFunction:
    """
    Class RayleighPoincareLossFunction.
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
    def __init__(self, basis, jac_u=None, jac_basis=None):
        self.basis = basis
        self.jac_u = jac_u
        self.jac_basis = jac_basis

    @staticmethod
    def _eval(G, jac_u, jac_basis):
        """
        Evaluate the Poincare based loss for the feature map G.T @ self.basis.

        Parameters
        ----------
        G : numpy.ndarray
            The coefficients of the feature map in the basis.
            Has shape (K, m).
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
        loss : float
            Poincare based loss.
        """
        jac_g = np.einsum("ij, kjl", G.T, jac_basis)
        jac_g = np.moveaxis(jac_g, 0, 1)
        return PoincareLossFunction._eval(jac_u, jac_g)

    def eval(self, G, jac_u=None, jac_basis=None):
        """
        Evaluate the Poincare based loss for the feature map G.T @ self.basis.
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        G : numpy.ndarray
            The coefficients of the feature map in the basis.
            Has shape (K, m).
        jac_u : numpy.ndarray, optional
            N Samples of the jacobian of the function to approximate.
            jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
            Has shape (N, K, d).
        jac_basis : numpy.ndarray, optional
            N Samples of the jacobian of the basis spanning the space of feature maps.
            jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
            Has shape (N, n, d).

        Returns
        -------
        loss : float
            Poincare based loss.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval(G, jac_u, jac_basis)

    @staticmethod
    def _eval_gradient(G, jac_u, jac_basis):
        m = G.shape[1]
        d = jac_u.shape[2]
        N = len(jac_u)
        jac_g = np.moveaxis(np.einsum("ij, kjl", G.T, jac_basis), 0, 1)
        gradients = np.zeros((N,) + G.shape)
        for k, jb, jg, ju in zip(np.arange(N), jac_basis, jac_g, jac_u):
            ug, sg, vgh = np.linalg.svd(jg)
            ug, vgh = ug[:, :m], vgh[:m, :]
            res1 = jb @ (np.eye(d) - vgh.T @ vgh) @ ju.T
            res2 = ug @ np.diag(1 / sg) @ vgh @ ju.T
            gradients[k] = - 2 * res1 @ res2.T
        gradient = gradients.mean(axis=0)
        return gradient

    def eval_gradient(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_gradient(G, jac_u, jac_basis)

    @staticmethod
    def _apply_sigma_h(X, G, jac_u, jac_basis):
        """
        Compute the matrix-vector multiplication Sigma(G)X and H(G)X as described in Bigoni et al. 2022, where
        X is a column-major vectorized matrix of same size as G.
        In other words, it computes Sigma(G) @ X.flatten('F') and H(G) @ X.flatten('F').

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        Sx : numpy.ndarray
            Has same shape as X.
        Hx : numpy.ndarray
            Has same shape as X.
        """
        if X.ndim == 1:
            Xmat = X.reshape(G.shape, order='F')  # column-major ordering
        else:
            Xmat = X
        N = len(jac_u)
        Hx = np.zeros(Xmat.shape)
        Sx = np.zeros(Xmat.shape)
        for jb, ju in zip(jac_basis, jac_u):
            jbju = jb @ ju.T
            jg = G.T @ jb
            jgju = jg @ ju.T
            GBG = jg @ jg.T
            GAG = jgju @ jgju.T
            Y = np.linalg.solve(GBG.T, Xmat.T)
            Hx += jbju @ jbju.T @ Y.T / N
            Sx += jb @ jb.T @ np.linalg.solve(GBG.T, GAG.T @ Y).T / N

        Sx = Sx.reshape(X.shape, order='F')  # column-major ordering
        Hx = Hx.reshape(X.shape, order='F')
        return Sx, Hx

    def apply_sigma_h(self, X, G, jac_u=None, jac_basis=None):
        """
        Compute the matrix-vector multiplication Sigma(G)X and H(G)X as described in Bigoni et al. 2022, where
        X is a column-major vectorized matrix of same size as G.
        In other words, it computes Sigma(G) @ X.flatten('F') and H(G) @ X.flatten('F').
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray, optional.
            Has shape (N, n, d)
        jac_basis : numpy.ndarray, optional.
            Has shape (N, K, d)

        Returns
        -------
        Sx : numpy.ndarray
            Has same shape as X.
        Hx : numpy.ndarray
            Has same shape as X.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_sigma_h(X, G, jac_u, jac_basis)

    @staticmethod
    def _eval_sigma_diag(G, jac_u, jac_basis):
        """
        Compute the diagnoal of the matrix Sigma(G) from Bigoni et al. 2022.
        It is used for preconditioning the conjugate gradient used to apply the inverse of Sigma(G).

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        N = len(jac_u)
        diag = np.zeros(G.shape[0] * G.shape[1])
        for jb, ju in zip(jac_basis, jac_u):
            B = jb @ jb.T
            A = jb @ ju.T @ ju @ jb.T
            GBG = G.T @ B @ G
            GAG = G.T @ A @ G
            M = np.linalg.solve(GBG.T, GAG.T).T
            M = np.linalg.solve(GBG, M)
            diag1 = np.diag(M)
            diag2 = np.diag(B)
            diag += np.kron(diag1, diag2) / N
        return diag

    def eval_sigma_diag(self, G, jac_u=None, jac_basis=None):
        """
        Compute the diagnoal of the matrix Sigma(G) from Bigoni et al. 2022.
        It is used for preconditioning the conjugate gradient used to apply the inverse of Sigma(G).
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_sigma_diag(G, jac_u, jac_basis)

    @staticmethod
    def _apply_sigma_inv(X, G, jac_u, jac_basis, **kwargs):
        """
        Apply the inverse of the matrix Sigma(G) from Bigoni et al. 2022 to a vector X.
        The conjugate gradient method is used with Jacobi preconditioning.

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)
        **kwargs : dict
            Key word arguments for scipy.sparse.linalg.cg.

        Returns
        -------
        out : numpy.ndarray
            Has same shape as X.
        """
        if X.ndim == 2:
            Xvec = X.flatten(order='F')  # column-major flattening
        else:
            Xvec = X
        Km = G.shape[0] * G.shape[1]
        matvec = lambda X: RayleighPoincareLossFunction._apply_sigma_h(X, G, jac_u, jac_basis)[0]
        sigma = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec)
        diag = RayleighPoincareLossFunction._eval_sigma_diag(G, jac_u, jac_basis)
        M = scipy.sparse.diags(1/diag)
        out, info = scipy.sparse.linalg.cg(sigma, Xvec, M=M, **kwargs)
        out = out.reshape(X.shape, order='F')
        return out

    def apply_sigma_inv(self, X, G, jac_u=None, jac_basis=None, **kwargs):
        """
        Apply the inverse of the matrix Sigma(G) from Bigoni et al. 2022 to a vector X.
        The conjugate gradient method is used with Jacobi preconditioning.
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)
        **kwargs : dict
            Key word arguments for scipy.sparse.linalg.cg.

        Returns
        -------
        out : numpy.ndarray
            Has same shape as X.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_sigma_inv(X, G, jac_u, jac_basis, **kwargs)

    @staticmethod
    def _eval_sigma_h_full(G, jac_u, jac_basis):
        """
        Build the full matrices Sigma(G) and H(G) from Bigoni et al. 2022.
        This should only be used for debugging or testing purpose.

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        S : numpy.ndarray
            Has shape (K*m, K*m).
        H : numpy.ndarray
            Has shape (K*m, K*m).
        """
        Km = G.shape[0] * G.shape[1]
        N = jac_u.shape[0]
        S = np.zeros((Km, Km))
        H = np.zeros((Km, Km))
        for ju, jb in zip(jac_u, jac_basis):
            jbju = jb @ ju.T
            A = jbju @ jbju.T
            B = jb @ jb.T
            jg = G.T @ jb
            jgju = jg @ ju.T
            GBG = jg @ jg.T
            GAG = jgju @ jgju.T
            GBG_inv = np.linalg.inv(GBG)
            S += np.kron(GBG_inv @ GAG @ GBG_inv, B) / N
            H += np.kron(GBG_inv, A) / N
        return S, H

    def eval_sigma_h_full(self, G, jac_u=None, jac_basis=None):
        """
        Build the full matrices Sigma(G) and H(G) from Bigoni et al. 2022.
        This should only be used for debugging or testing purpose.
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray, optional
            Has shape (N, n, d)
        jac_basis : numpy.ndarray, optional
            Has shape (N, K, d)

        Returns
        -------
        S : numpy.ndarray
            Has shape (K*m, K*m).
        H : numpy.ndarray
            Has shape (K*m, K*m).
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_sigma_h_full(G, jac_u, jac_basis)

    @staticmethod
    def _apply_hessian(X, G, jac_u, jac_basis):
        """
        Compute the matrix-vector multiplication Hess(G) @ X where
        X is a column-major vectorized matrix of same size as G.
        In other words, it computes Hess(G) @ X.flatten('F').

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray, optional.
            Has shape (N, n, d)
        jac_basis : numpy.ndarray, optional.
            Has shape (N, K, d)

        Returns
        -------
        hess : numpy.ndarray
            Has same shape as X.
        """
        if X.ndim == 1:
            Xmat = X.reshape(G.shape, order='F')  # column-major ordering
        else:
            Xmat = X
        d = jac_u.shape[-1]
        N = len(jac_u)
        hess = 0 * Xmat
        for jb, ju in zip(jac_basis, jac_u):
            grad_u = ju.T
            djg = Xmat.T @ jb
            jg = G.T @ jb
            jg_inv = scipy.linalg.pinv(jg)
            p = np.eye(d) - jg_inv @ jg
            res1 = 2 * jg_inv.T @ (djg.T @ jg_inv.T - jg_inv @ djg @ p) @ grad_u @ grad_u.T @ p
            res2 = 2 * jg_inv.T @ grad_u @ grad_u.T @ (jg_inv @ djg @ p + p @ djg.T @ jg_inv.T)
            hess += jb @ (res1 + res2).T / N

        hess = hess.reshape(X.shape, order='F')  # column-major ordering
        return hess

    def apply_hessian(self, X, G, jac_u=None, jac_basis=None):
        """
        Compute the matrix-vector multiplication Hess(G) @ X where
        X is a column-major vectorized matrix of same size as G.
        In other words, it computes Hess(G) @ X.flatten('F').

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray, optional.
            Has shape (N, n, d)
        jac_basis : numpy.ndarray, optional.
            Has shape (N, K, d)

        Returns
        -------
        hess : numpy.ndarray
            Has same shape as X.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_hessian(X, G, jac_u, jac_basis)

    @staticmethod
    def _eval_hessian_diag(G, jac_u, jac_basis):
        """
        Compute the diagnoal of the matrix Hess(G).
        It is used for preconditioning the conjugate gradient used to apply the inverse of Hess(G).
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        d = jac_u.shape[-1]
        N = len(jac_u)
        diag_mat = 0 * G
        for jb, ju in zip(jac_basis, jac_u):
            grad_u = ju.T
            jg = G.T @ jb
            jg_inv = scipy.linalg.pinv(jg)
            p = np.eye(jb.shape[1]) - jg_inv @ jg
            diag_mat += (jb @ p @ grad_u @ grad_u.T @ jg_inv) * (jb @ jg_inv)
            diag_mat -= (jb @ p @ grad_u) ** 2 @ np.ones((ju.shape[0], 1)) @ np.diag(jg_inv.T @ jg_inv).reshape(1, -1)
            diag_mat += (jb @ jg_inv) * (jb @ p @ grad_u @ grad_u.T @ jg_inv)
            diag_mat += (jb @ p) ** 2 @ np.ones((d, 1)) @ np.diag(jg_inv.T @ grad_u @ grad_u.T @ jg_inv).reshape(1, -1)
        diag_mat = 2 * diag_mat / N
        diag = diag_mat.flatten(order="F")
        return diag

    def eval_hessian_diag(self, G, jac_u=None, jac_basis=None):
        """
        Compute the diagnoal of the matrix Hess(G).
        It is used for preconditioning the conjugate gradient used to apply the inverse of Hess(G).
        If optional parameters are not provided, then the corresponding attributes are used.

        Parameters
        ----------
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)

        Returns
        -------
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_hessian_diag(G, jac_u, jac_basis)

    @staticmethod
    def _apply_hessian_inv(X, G, jac_u, jac_basis, **kwargs):
        """
        Apply the inverse of the matrix Hess(G) to a vector X.
        The conjugate gradient method is used with Jacobi preconditioning.

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)
        **kwargs : dict
            Key word arguments for scipy.sparse.linalg.cg.

        Returns
        -------
        out : numpy.ndarray
            Has same shape as X.
        """
        if X.ndim == 2:
            Xvec = X.flatten(order='F')  # column-major flattening
        else:
            Xvec = X
        Km = G.shape[0] * G.shape[1]
        matvec = lambda X: RayleighPoincareLossFunction._apply_hessian(X, G, jac_u, jac_basis)
        hess = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec)
        diag = RayleighPoincareLossFunction._eval_hessian_diag(G, jac_u, jac_basis)
        M = scipy.sparse.diags(1 / diag)
        out, info = scipy.sparse.linalg.cg(hess, Xvec, M=M, **kwargs)
        out = out.reshape(X.shape, order='F')
        return out

    def apply_hessian_inv(self, X, G, jac_u=None, jac_basis=None, **kwargs):
        """
        Apply the inverse of the matrix Hess(G) to a vector X.
        The conjugate gradient method is used with Jacobi preconditioning.

        Parameters
        ----------
        X : numpy.ndarray
            Has shape (K, m) or (K*m, ).
        G : numpy.ndarray
            Has shape (K, m).
        jac_u : numpy.ndarray
            Has shape (N, n, d)
        jac_basis : numpy.ndarray
            Has shape (N, K, d)
        **kwargs : dict
            Key word arguments for scipy.sparse.linalg.cg.

        Returns
        -------
        out : numpy.ndarray
            Has same shape as X.
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_hessian_inv(X, G, jac_u, jac_basis, **kwargs)
    