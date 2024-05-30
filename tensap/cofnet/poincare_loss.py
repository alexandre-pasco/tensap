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
    def _apply_s_h(X, G, jac_u, jac_basis):
        """
        ...

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
            Xmat = X.reshape(G.shape, order='F')  # column-wise ordering
        else:
            Xmat = X
        N = len(jac_u)
        Hx = np.zeros(Xmat.shape)
        Sx = np.zeros(Xmat.shape)
        for jb, ju in zip(jac_basis, jac_u):
            jbju = jb @ ju.T
            jg = G.T @ jb
            jgju = jg @ ju.T
            # B = jb @ jb.T
            # A = jb @ ju.T @ ju @ jb.T
            GBG = jg @ jg.T
            GAG = jgju @ jgju.T
            Y = np.linalg.solve(GBG.T, Xmat.T)
            Hx += jbju @ jbju.T @ Y.T / N
            Sx += jb @ jb.T @ np.linalg.solve(GBG.T, GAG.T @ Y).T / N

        if X.ndim == 1:
            Sx = Sx.flatten(order='F')  # column-wise ordering
            Hx = Hx.flatten(order='F')

        return Sx, Hx

    def apply_s_h(self, X, G, jac_u=None, jac_basis=None):
        """
        ...

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
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_s_h(X, G, jac_u, jac_basis)

    @staticmethod
    def _build_operator_s_h(G, jac_u, jac_basis):
        """
        ...

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
        S : scipy.sparse.linalg.LinearOperator
            Has shape (K*m, K*m).
        H : scipy.sparse.linalg.LinearOperator
            Has shape (K*m, K*m).
        """
        Km =  G.shape[0] * G.shape[1]
        matvec_s = lambda X : RayleighPoincareLossFunction._apply_s_h(X, G, jac_u, jac_basis)[0]
        matvec_h = lambda X: RayleighPoincareLossFunction._apply_s_h(X, G, jac_u, jac_basis)[1]
        S = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec_s)
        H = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec_h)
        return S, H

    def build_operator_s_h(self, G, jac_u=None, jac_basis=None):
        """
        ...

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
        S : scipy.sparse.linalg.LinearOperator
            Has shape (K*m, K*m).
        H : scipy.sparse.linalg.LinearOperator
            Has shape (K*m, K*m).
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_operator_s_h(G, jac_u, jac_basis)

    @staticmethod
    def _build_diag_s(G, jac_u, jac_basis):
        """
        ...
        Compute the diagnoal of the matrix Sigma from Bigoni et al. 2022.

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

    def build_diag_s(self, G, jac_u=None, jac_basis=None):
        """
        ...
        Compute the diagnoal of the matrix Sigma from Bigoni et al. 2022.

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
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_diag_s(G, jac_u, jac_basis)

    @staticmethod
    def _apply_inv_s(X, G, jac_u, jac_basis, **kwargs):
        """
        ...
        Compute the diagnoal of the matrix Sigma from Bigoni et al. 2022.

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
        diag : numpy.ndarray
            Has shape (K*m, ).
        """
        #TODO when G is flattened or not
        sigma = RayleighPoincareLossFunction._build_operator_s_h(G, jac_u, jac_basis)[0]
        diag = RayleighPoincareLossFunction._build_diag_s(G, jac_u, jac_basis)
        M = scipy.sparse.diags(1/diag)
        out = scipy.sparse.linalg.cg(sigma, X, M=M, **kwargs)[0]
        return out

    def apply_inv_s(self, X, G, jac_u=None, jac_basis=None, **kwargs):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_inv_s(X, G, jac_u, jac_basis, **kwargs)

    @staticmethod
    def _build_full_s_h(G, jac_u, jac_basis):
        """
        ...

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

    def build_full_s_h(self, G, jac_u=None, jac_basis=None):
        """
        ...

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
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_full_s_h(G, jac_u, jac_basis)
