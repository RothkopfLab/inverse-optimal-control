import jax.ops
import numpy as np
import jax.numpy as jnp
from ioc.examples.problem import ControlTask, PartiallyObservableControlTask


class ReachingProblem(ControlTask):
    def __init__(self, c=-0.3, r=-5, T=30, v=-0.69897, f=-1.69897, target_dist=0.1, dt=0.01):
        # setup problem
        m = 1  # mass(kg)
        b = 0  # damping(N / sec)
        tau = 40  # time constant(msec)
        pos = 0.5 * 0.02  # position noise
        vel = 0.5 * 0.2  # velocity noise
        frc = 0.5 * 1.0  # force noise
        T = T  # duration in number of time steps

        # compute system dynamics and cost matrices
        dtt = dt / (tau / 1000)
        A = np.zeros((5, 5))
        A[0, 0] = 1
        A[0, 1] = dt
        A[1, 1] = 1 - dt * b / m
        A[1, 2] = dt / m
        A[2, 2] = 1 - dtt
        A[2, 3] = dtt
        A[3, 3] = 1 - dtt
        A[4, 4] = 1
        A = jnp.array(A)

        B = np.zeros((5, 1))
        B[3, 0] = dtt
        B = jnp.array(B)

        C = 10 ** c
        C0 = jnp.eye(5) * 0.

        H = np.zeros((3, 5))
        H[0:3, 0:3] = np.eye(3)
        H = jnp.array(H)

        D = 0.
        D0 = jnp.diag(jnp.array([pos, vel, frc]))

        E0 = jnp.eye(5) * 0.

        R = 10 ** r / T

        # final time step costs
        d = jnp.block([[1., 0., 0., 0., -1.],
                       [0., 10 ** v, 0., 0., 0.],
                       [0., 0., 10 ** f, 0., 0.]])
        Q = jnp.zeros((5, 5, T))
        Q = jax.ops.index_update(Q, jax.ops.index[:, :, -1], d.T @ d)

        x0 = np.zeros((5, 1))
        x0[4] = target_dist
        S0 = jnp.eye(5) * 0.

        B0 = jnp.array(x0)
        V0 = jnp.eye(5) * 1e-1

        super().__init__(A, B, C, C0, H, D, D0, E0, Q, R, x0, S0, B0=B0, V0=V0)


class PartiallyObservedReachingProblem(PartiallyObservableControlTask):
    def __init__(self, c=-.3, r=-5, T=30, v=-0.69897, f=-1.69897, obs_noise=1e-5, target_dist=0.1, S0=None):
        # setup problem
        dt = 0.01  # time step (sec)
        m = 1  # mass(kg)
        b = 0  # damping(N / sec)
        tau = 40  # time constant(msec)
        pos = 0.5 * 0.02  # position noise
        vel = 0.5 * 0.2  # velocity noise
        frc = 0.5 * 1.0  # force noise
        T = T  # duration in number of time steps

        # compute system dynamics and cost matrices
        dtt = dt / (tau / 1000)
        A = np.zeros((5, 5))
        A[0, 0] = 1
        A[0, 1] = dt
        A[1, 1] = 1 - dt * b / m
        A[1, 2] = dt / m
        A[2, 2] = 1 - dtt
        A[2, 3] = dtt
        A[3, 3] = 1 - dtt
        A[4, 4] = 1
        A = jnp.array(A)

        B = np.zeros((5, 1))
        B[3, 0] = dtt
        B = jnp.array(B)

        C = 10 ** c
        C0 = jnp.eye(5) * 0.

        H = np.zeros((3, 5))
        H[0:3, 0:3] = np.eye(3)
        H = jnp.array(H)

        D = 0.
        D0 = jnp.diag(jnp.array([pos, vel, frc]))

        E0 = jnp.eye(5) * 0.

        R = 10 ** r / T

        d = jnp.block([[1., 0., 0., 0., -1.],
                       [0., 10 ** v, 0., 0., 0.],
                       [0., 0., 10 ** f, 0., 0.]])
        Q = jnp.zeros((5, 5, T))
        Q = jax.ops.index_update(Q, jax.ops.index[:, :, -1], d.T @ d)

        x0 = np.zeros((5, 1))
        x0[4] = target_dist
        if S0 is None:
            S0 = jnp.eye(5) * 0.

        B0 = jnp.array(x0)
        V0 = jnp.eye(5) * 0.

        # experimenter's observation model
        S = jnp.array([[1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.]])
        U = jnp.eye(2) * obs_noise

        super().__init__(A, B, C, C0, H, D, D0, E0, Q, R, x0, S0, B0=B0, V0=V0, S=S, U=U)
