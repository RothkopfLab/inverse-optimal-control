import jax.numpy as jnp


class ControlTask:
    def __init__(self, A, B, C, C0, H, D, D0, E0, Q, R, X0, S0, B0=None, V0=None):
        self.A = A  # plant dynamics
        self.B = B  # control dynamics
        self.H = H  # feedback state transformation
        xdim = self.xdim
        udim = self.udim
        ydim = self.ydim

        # expand possibly scalar variables
        C = jnp.array(C)
        D = jnp.array(D)
        C0 = jnp.array(C0)
        D0 = jnp.array(D0)
        E0 = jnp.array(E0)
        R = jnp.array(R).reshape((udim, udim))
        if C.size == 1:
            C = jnp.broadcast_to(C, (udim, udim, 1))
        if D.size == 1:
            D = jnp.broadcast_to(D, (ydim, xdim, 1))
        if C0.size == 1:
            C0 = jnp.broadcast_to(C0, (xdim, 1))
        if D0.size == 1:
            D0 = jnp.broadcast_to(D0, (ydim, 1))
        if E0.size == 1:
            E0 = jnp.broadcast_to(E0, (xdim, 1))

        self.C = C  # action dependent noise transform in dynamics (of shape U x U x NumNoiseTerms)
        self.D = D  # state dependent noise transform in feedback (of shape Y x X x NumNoiseTerms)

        self.Q = Q  # state cost xQx.T (of shape X x X)
        self.R = R  # action cost uRu.T (of shape U x U)

        self.C0 = C0  # cholesky decomposition of dynamics noise (xi) covariance (C0 @ C0.T of shape XxX)
        self.D0 = D0  # cholesky decomposition of feedback noise (omega) covariance (D0 @ D0.T of shape YxY)
        self.E0 = E0  # cholesky decomposition of estimator noise (eta) covariance (E0 @ E0.T of shape XxX)

        self.X0 = X0  # starting state
        self.S0 = S0  # starting sigma (uncertainty of state estimate)

        self.S = jnp.eye(xdim)
        self.U = jnp.zeros((xdim, xdim))

        self.B0 = B0 if B0 is not None else X0  # initial mean belief of the state
        self.V0 = V0 if V0 is not None else jnp.eye(self.xdim) * 1e-7  # initial variance belief of the state

        self.T = Q.shape[2]

    @property
    def C02(self):
        return self.C0 @ self.C0.T

    @property
    def D02(self):
        return self.D0 @ self.D0.T

    @property
    def E02(self):
        return self.E0 @ self.E0.T

    @property
    def xdim(self):
        return self.A.shape[0]

    @property
    def udim(self):
        return self.B.shape[1]

    @property
    def ydim(self):
        return self.H.shape[0]

    @property
    def zdim(self):
        return self.S.shape[0]


class FullyObservableControlTask(ControlTask):
    pass


class PartiallyObservableControlTask(ControlTask):
    def __init__(self, A, B, C, C0, H, D, D0, E0, Q, R, X0, S0, B0=None, V0=None, S=None, U=None):
        super().__init__(A, B, C, C0, H, D, D0, E0, Q, R, X0, S0, B0=B0, V0=V0)

        self.S = jnp.eye(self.xdim) if S is None else S
        self.U = jnp.eye(self.xdim) * 0. if U is None else U
