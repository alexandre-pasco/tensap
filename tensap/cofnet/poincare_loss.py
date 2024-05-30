import numpy as np
import scipy


class PoincareLossFunction:
    """
    Class PoincareLossFunction.

    Attributes
    ----------
    error_type : string, optional
        The error type. The default is 'relative'. Can also be 'absolute'.

    """

    def __init__(self, jac_u=None, jac_g=None):
        """
        Constructor for the class PoincareLossFunction.

        Returns
        -------
        None.

        """
        self.jac_u = jac_u
        self.jac_g = jac_g

    @staticmethod
    def _eval(jac_u, jac_g):
        jac_ut = np.moveaxis(jac_u, -1, -2)
        proj = np.array([jg.T @ np.linalg.solve(jg @ jg.T, jg) for jg in jac_g])
        residual = jac_ut - np.einsum("kil,klj->kij", proj, jac_ut, optimize=True)
        losses = np.linalg.norm(residual, axis=(1,2)) ** 2
        loss = losses.mean()
        return loss

    def eval(self, jac_u=None, jac_g=None):
        if jac_u is None : jac_u = self.jac_u
        if jac_g is None : jac_g = self.jac_g
        return self._eval(self.jac_u, self.jac_g)


class RayleighPoincareLossFunction:

    def __init__(self, jac_u=None, jac_basis=None):
        self.jac_u = jac_u
        self.jac_basis = jac_basis

    def build_jac_g(self, G):
        def jac_g(x):
            out = np.einsum("ij, kjl", G.T, self.basis.eval_jacobian(x))
            return np.moveaxis(out, 0, 1)
        return jac_g

    @staticmethod
    def _eval(G, jac_u, jac_basis):
        jac_g = np.einsum("ij, kjl", G.T, jac_basis)
        jac_g = np.moveaxis(jac_g, 0, 1)
        return PoincareLossFunction._eval(jac_u, jac_g)

    def eval(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval(G, jac_u, jac_basis)

    @staticmethod
    def _eval_gradient(G, jac_u, jac_basis):
        return RayleighPoincareLossFunction._eval_gradients(G, jac_u, jac_basis).mean(axis=0)

    def eval_gradient(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_gradient(G, jac_u, jac_basis)

    @staticmethod
    def _eval_gradients(G, jac_u, jac_basis):
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
        return gradients

    def eval_gradients(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._eval_gradients(G, jac_u, jac_basis)

    @staticmethod
    def _apply_s_h(X, G, jac_u, jac_basis):
        if X.ndim == 1:
            Xmat = X.reshape(G.shape, order='F')  # column-wise oredering
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
            Sx = Sx.flatten(order='F') # column-wise ordering
            Hx = Hx.flatten(order='F')

        return Sx, Hx

    def apply_s_h(self, X, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._apply_s_h(X, G, jac_u, jac_basis)

    @staticmethod
    def _build_operator_s_h(G, jac_u, jac_basis):
        Km =  G.shape[0] * G.shape[1]
        matvec_s = lambda X : RayleighPoincareLossFunction._apply_s_h(X, G, jac_u, jac_basis)[0]
        matvec_h = lambda X: RayleighPoincareLossFunction._apply_s_h(X, G, jac_u, jac_basis)[1]
        S = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec_s)
        H = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec_h)
        return S, H

    def build_operator_s_h(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_operator_s_h(G, jac_u, jac_basis)

    @staticmethod
    def _build_diag_s(G, jac_u, jac_basis):
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
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_diag_s(G, jac_u, jac_basis)

    @staticmethod
    def _apply_inv_s(X, G, jac_u, jac_basis, **kwargs):
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
    def _build_full_s(G, jac_u, jac_basis):
        Km = G.shape[0] * G.shape[1]
        N = jac_u.shape[0]
        S = np.zeros((Km, Km))
        for ju, jb in zip(jac_u, jac_basis):
            B = jb @ jb.T
            jg = G.T @ jb
            jgju = jg @ ju.T
            GBG = jg @ jg.T
            GAG = jgju @ jgju.T
            GBG_inv = np.linalg.inv(GBG)
            S += np.kron(GBG_inv @ GAG @ GBG_inv, B) / N
        return S

    def build_full_s(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_full_s(G, jac_u, jac_basis)

    @staticmethod
    def _build_full_h(G, jac_u, jac_basis):
        Km = G.shape[0] * G.shape[1]
        N = jac_u.shape[0]
        H = np.zeros((Km, Km))
        for ju, jb in zip(jac_u, jac_basis):
            B = jb @ jb.T
            jbju = jb @ ju.T
            A = jbju @ jbju.T
            GBG = G.T @ jb @ jb.T @ G
            GBG_inv = np.linalg.inv(GBG)
            H += np.kron(GBG_inv, A) / N
        return H

    def build_full_h(self, G, jac_u=None, jac_basis=None):
        if jac_u is None: jac_u = self.jac_u
        if jac_basis is None: jac_basis = self.jac_basis
        return self._build_full_h(G, jac_u, jac_basis)
