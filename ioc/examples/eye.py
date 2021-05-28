import jax.ops
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import scipy.linalg as linalg

from ioc.examples.problem import ControlTask


class SaccadeModel(ControlTask):
    """ Saccade model from Crevecoeur & Kording (2016) """

    def __init__(self, T=40, r=-4., tf=10, ti=5):
        # parameters
        dt = 1.25  # time step (msec)
        c0 = 1e-2
        alpha = jnp.sqrt(.08)
        target_dist = 10.
        sigma = 1e-3

        # time constants
        tau1 = 224
        tau2 = 13
        tau_prod = tau1 * tau2

        # compute discretized system dynamics
        A0 = np.array([[0., 1.], [-1 / tau_prod, -(tau1 + tau2) / (tau_prod)]])
        As = linalg.expm(dt * A0)
        A = jlinalg.block_diag(As, 1.)

        xdim = A.shape[0]

        B0 = np.array([0, 1 / tau_prod])[:, None]

        # Taylor series approximation of B matrix
        taylor = dt * np.array(
            [np.linalg.matrix_power(A0 * dt, k) / float(np.math.factorial(k + 1)) for k in range(50)])
        Bs = taylor.sum(axis=0) @ B0
        B = np.vstack([Bs, 0.])

        # control-dependent noise
        C = alpha
        C0 = np.sqrt(np.diag(np.diag(c0 * (Bs @ Bs.T))))
        C0 = linalg.block_diag(C0, 0.)

        H = jnp.eye(2, xdim)

        # convert to jax
        A = jnp.array(A)
        B = jnp.array(B)
        C0 = jnp.array(C0)

        # observation noise
        D = 0.
        D0 = jnp.eye(2) * sigma

        # belief noise
        E0 = jnp.eye(xdim) * 0.

        # control cost
        R = 10 ** r / T

        # Initial and final cost functions
        d1 = np.array([[1., 0., 1.]])
        d2 = jnp.array([[1., 0., -1.]])

        Q = jnp.zeros((xdim, xdim, T))
        Q = jax.ops.index_update(Q, jax.ops.index[:, :, :ti], (d1.T @ d1)[..., None])
        Q = jax.ops.index_update(Q, jax.ops.index[:, :, -tf:], (d2.T @ d2)[..., None])

        # initialization of state and belief
        x0 = jnp.array([[-target_dist], [0.], [target_dist]])
        S0 = jnp.diag(jnp.array([0., 0., 0.]))

        B0 = jnp.array(x0)
        V0 = jnp.eye(xdim) * 1e-3

        super().__init__(A, B, C, C0, H, D, D0, E0, Q, R, x0, S0, B0=B0, V0=V0)
