import numpy as np
import jax.numpy as jnp
import jax.lax as jlax
import jax.ops as jops
import jax.scipy.linalg as jlinalg
from jax import vmap, random
from enum import Enum
from abc import ABC

quadratic_form = vmap(lambda A, B: A @ B @ A.T, in_axes=(0, None))


class KalmanInit(Enum):
    OPEN_LOOP = 1
    RANDOM = 2


class Solver(ABC):
    def __init__(self, control_prob):
        self.cp = control_prob

        self.L = None
        self.K = None

        self.rng = random.PRNGKey(0)

    def avg_trajectory(self):
        # compute avg trajectory
        T = self.cp.T
        Xa = np.zeros((self.cp.xdim, T))
        Xa[:, 0] = np.squeeze(self.cp.X0)

        for k in range(T - 1):
            u = -self.L[k, :, :] @ Xa[:, k]
            Xa[:, k + 1] = self.cp.A @ Xa[:, k] + self.cp.B @ u

        return Xa

    def simulate_trajectories(self, num_trajectories, return_x_sim=False):
        A = self.cp.A
        B = self.cp.B
        C = self.cp.C
        H = self.cp.H

        D = self.cp.D
        Q = self.cp.Q
        R = self.cp.R
        T = self.cp.T

        C0 = self.cp.C0
        D0 = self.cp.D0
        E0 = self.cp.E0

        xdim = self.cp.xdim
        udim = self.cp.udim
        ydim = self.cp.ydim
        zdim = self.cp.zdim

        cost_sim = 0.
        XSim = np.zeros((xdim, num_trajectories, T))
        Xhat = np.zeros((xdim, num_trajectories, T))
        XObs = np.zeros((zdim, num_trajectories, T))
        Us = np.zeros((udim, num_trajectories, T - 1))

        # simulate noisy trajectories
        if num_trajectories > 0:
            # init
            XSim[:, :, 0] = self.cp.X0 + self.cp.S0 @ np.random.normal(0, 1, size=(xdim, num_trajectories))
            Xhat[:, :, 0] = XSim[:, :, 0] + self.cp.S0 @ np.random.normal(0, 1, size=(xdim, num_trajectories))
            XObs[:, :, 0] = self.cp.S @ XSim[:, :, 0] + self.cp.U @ np.random.normal(0, 1, size=(
                self.cp.U.shape[1], num_trajectories))

            for k in range(T - 1):
                # compute control
                U = -self.L[k, :, :] @ Xhat[:, :, k]
                Us[:, :, k] = U

                # compute cost
                cost_sim += np.sum(U * (R @ U))
                cost_sim += np.sum(XSim[:, :, k] * (Q[:, :, k] @ XSim[:, :, k]))

                # compute noisy control
                CU = np.sum(C[:, :, None, :] * U[None, :, :, None], axis=1)
                Un = U + np.sum(CU * np.random.normal(0, 1, size=(1, num_trajectories, CU.shape[2])), axis=2)

                # compute noisy observation
                y = H @ XSim[:, :, k] + D0 @ np.random.normal(0, 1, size=(D0.shape[1], num_trajectories))
                DXSim = np.sum(D[:, :, None, :] * XSim[:, :, k, None], axis=1)
                y += np.sum(DXSim * np.random.normal(0, 1, size=(1, num_trajectories, D.shape[2])), axis=2)

                XSim[:, :, k + 1] = A @ XSim[:, :, k] + B @ Un + C0 @ np.random.normal(0, 1, size=(
                    C0.shape[1], num_trajectories))
                Xhat[:, :, k + 1] = self.apply_filter(Xhat[:, :, k], U, y, k)

                # add observation noise
                XObs[:, :, k + 1] = self.cp.S @ XSim[:, :, k + 1] + self.cp.U @ np.random.normal(0, 1, size=(
                    self.cp.U.shape[1], num_trajectories))

            cost_sim += np.sum(XSim[:, :, -1] * (Q[:, :, -1] @ XSim[:, :, -1]))
            cost_sim /= num_trajectories

        if return_x_sim:
            return XObs, cost_sim, XSim, Xhat, Us
        else:
            return XObs, cost_sim

    def apply_filter(self, x_hat, U, y, t):
        num_trajectories = x_hat.shape[1]
        x_next_hat = self.cp.A @ x_hat + self.cp.B @ U + self.K[t, :, :] @ (
                y - self.cp.H @ x_hat) + self.cp.E0 @ np.random.normal(0, 1,
                                                                       size=(self.cp.E0.shape[1], num_trajectories))
        return x_next_hat


class TodorovSOC(Solver):
    def __init__(self, control_prob):
        super().__init__(control_prob)

    # forward pass / filter (5.2 in the paper)
    @staticmethod
    def forward_pass(A, B, C, H, D, C02, D02, E02, S0, X0, L):
        # initialize covariances
        xdim = S0.shape[0]

        # lax scan
        # input Lt, output Kt

        # state: Se, Sx, cost
        state_0 = {
            'SiX': X0 @ X0.T,
            'SiE': S0,
            'SiXE': jnp.zeros((xdim, xdim)),
        }

        # for t in range(T - 1):
        def forward_loop(state, Lt):
            SiX = state['SiX']
            SiE = state['SiE']
            SiXE = state['SiXE']

            # compute Kalman gain
            # Lt = L[:, :, t]

            # update K_t
            sum_S = SiE + SiX + SiXE + SiXE.T
            DSiD = quadratic_form(D.transpose((2, 0, 1)), sum_S).sum(axis=0)

            # Kt = A @ SiE @ H.T @ jlinalg.inv(H @ SiE @ H.T + D02 + DSiD)  # TODO unstable
            Kt = jlinalg.solve((H @ SiE @ H.T + D02 + DSiD).T, (A @ SiE @ H.T).T).T

            # compute Sigma_e
            LSiL = Lt @ SiX @ Lt.T
            next_SiE = C02 + E02 + (A - Kt @ H) @ SiE @ A.T
            # DIFFERENT FROM PAPER: B is multiplied with noise, too
            next_SiE += B @ quadratic_form(C.transpose((2, 0, 1)), LSiL).sum(axis=0) @ B.T

            # update Sigmas
            SiX = E02 + Kt @ H @ SiE @ A.T + (A - B @ Lt @ SiX @ (A - B @ Lt).T) + \
                  (A - B @ Lt @ SiXE @ H.T @ Kt.T + Kt @ H @ SiXE.T @ (A - B @ Lt).T)
            SiE = next_SiE
            SiXE = (A - B @ Lt) @ SiXE @ (A - Kt @ H).T - E02

            # define state for next iteration
            state = {
                'SiX': SiX,
                'SiE': SiE,
                'SiXE': SiXE
            }

            return state, Kt

        state, K = jlax.scan(forward_loop, state_0, L, reverse=False)
        return K

    # backward pass: compute optimal control (4.2 in the paper)
    @staticmethod
    def backward_pass(A, B, C, C02, D, D02, E02, H, T, Q, R, S0, X0, K, full_obs=False):
        xdim = A.shape[0]
        udim = B.shape[1]

        # lax scan
        # input that is scanned over: K and Q, output: L
        lin = {
            'K': K,
            'Q': Q[:-1, :, :]
        }

        # state: Se, Sx, cost
        state_0 = {
            'Sx': Q[-1, :, :],
            'Se': jnp.zeros((xdim, xdim)),
            'cost': 0.
        }

        def backward_loop(state, lin):
            # get states
            Sx = state['Sx']
            Se = state['Se']
            cost = state['cost']

            # get inputs
            # Kt = K[:, :, t]
            Kt = lin['K']
            # Qt = Q[:, :, t]
            Qt = lin['Q']

            # update cost
            cost += jnp.trace(Sx @ C02) + jnp.trace(Se @ (Kt @ D02 @ Kt.T + E02 + C02))

            # compute controller
            Lpinv = R + B.T @ Sx @ B

            if full_obs:
                BSxeB = B.T @ Sx @ B
            else:
                BSxeB = B.T @ (Sx + Se) @ B

            CSC = quadratic_form(C.transpose((2, 1, 0)), BSxeB).sum(axis=0)
            Lt = jlinalg.solve(Lpinv + CSC, B.T @ Sx @ A)

            newSe = A.T @ Sx @ B @ Lt + (A - Kt @ H).T @ Se @ (A - Kt @ H)
            Sx = Qt + A.T @ Sx @ (A - B @ Lt)
            KSeK = Kt.T @ Se @ Kt

            if not full_obs:
                Sx += quadratic_form(D.transpose((2, 1, 0)), KSeK).sum(axis=0)
            Se = newSe

            # define state for next iteration
            state = {
                'Sx': Sx,
                'Se': Se,
                'cost': cost
            }

            return state, Lt

        state, L = jlax.scan(backward_loop, state_0, lin, reverse=True)
        Sx = state['Sx']
        Se = state['Se']
        cost = state['cost']
        cost += jnp.squeeze(X0.T @ Sx @ X0) + jnp.trace((Se + Sx) * S0)

        return L, cost

    @staticmethod
    def run_solver(A, B, C, C02, D, D02, E02, H, T, Q, R, S0, X0, K0, max_iter, eps):
        costs = jnp.repeat(jnp.nan, max_iter + 1)

        if eps:

            def cond_fun(args):
                _, _, costs, i = args
                return (i <= 1) | ((i < max_iter) & (jnp.abs(costs[i - 1] - costs[i - 2]) >= eps))

            def body_fun(args):
                K, _, costs, i = args
                L, c = TodorovSOC.backward_pass(A, B, C, C02, D, D02, E02, H, T, Q, R, S0, X0, K)
                K_new = TodorovSOC.forward_pass(A, B, C, H, D, C02, D02, E02, S0, X0, L)
                costs = jops.index_update(costs, i, c)

                return [K_new, L, costs, i + 1]

            L_0 = jnp.zeros((T - 1, B.shape[1], A.shape[0]))
            (K, L, costs, i) = jlax.while_loop(cond_fun, body_fun, [K0, L_0, costs, 0])

        else:  # keep version without thresholds, to allow for backward-diff via jax.grad, just in case
            def body_fun(args, i):
                K, _ = args
                L, c = TodorovSOC.backward_pass(A, B, C, C02, D, D02, E02, H, T, Q, R, S0, X0, K)
                K_new = TodorovSOC.forward_pass(A, B, C, H, D, C02, D02, E02, S0, X0, L)

                return [K_new, L], c

            L_0 = jnp.zeros((T - 1, B.shape[1], A.shape[0]))
            (K, L), costs = jlax.scan(body_fun, [K0, L_0], jnp.arange(max_iter))

        # if i < max_iter:
        #     costs = costs[:i]

        return K, L, costs

    # Kalman LQG method of Todorov
    def run(self, init=KalmanInit.OPEN_LOOP, max_iter=500, eps=1e-14):
        A = self.cp.A
        B = self.cp.B
        C = self.cp.C
        H = self.cp.H

        D = self.cp.D
        Q = self.cp.Q.transpose((2, 0, 1))
        R = self.cp.R

        X0 = self.cp.X0
        S0 = self.cp.S0
        T = self.cp.T

        C02 = self.cp.C02
        D02 = self.cp.D02
        E02 = self.cp.E02

        xdim = self.cp.xdim
        udim = self.cp.udim
        ydim = self.cp.ydim

        if init == KalmanInit.OPEN_LOOP:
            K0 = jnp.zeros((T - 1, xdim, ydim))
        elif init == KalmanInit.RANDOM:
            K0 = random.uniform(self.rng, shape=(T - 1, xdim, ydim))

        K, L, c = TodorovSOC.run_solver(A, B, C, C02, D, D02, E02, H, T, Q, R, S0, X0, K0, max_iter, eps)

        self.L = L
        self.K = K

        return c
