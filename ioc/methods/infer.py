import jax
from jax.lax import scan
import jax.numpy as jnp
import numpyro.distributions as dist
from abc import ABC, abstractmethod

from ioc.examples.problem import PartiallyObservableControlTask
from ioc.methods.solvers import TodorovSOC

# batch outer product
vector_square = jax.vmap(lambda x: jnp.outer(x, x), 0, 0)

# quadratic form ABA'
quadratic_form = jax.vmap(lambda A, B: A @ B @ A.T, (None, 0), 0)


class ApproximateInference(ABC):

    def __init__(self, cp, solver=None):
        self.cp = cp
        self.solver = solver

    def log_likelihood(self, x, max_iter=50, eps=1e-14):
        """

        Args:
            cp: control problem instance
            x: (d x n x N) observed trajectories in n trials with d dimensions and N timesteps

        Returns:
            the log likelihood p(x | theta)
        """
        return self.conditional_distribution(x, max_iter, eps=eps).log_prob(x.transpose((2, 1, 0))[1:]).sum()

    @abstractmethod
    def conditional_moments(self, obs, max_iter=50, eps=1e-14):
        pass

    @abstractmethod
    def conditional_distribution(self, obs, max_iter=50, eps=1e-14):
        pass

    def solve_problem(self, max_iter=200, eps=1e-14):
        soc = TodorovSOC(self.cp)
        soc.run(max_iter=max_iter, eps=eps)
        self.solver = soc
        return soc


class FullyObservableApproximateInference(ApproximateInference):
    def conditional_distribution(self, z, max_iter=50, eps=1e-14):
        """

        Args:
            cp: control problem instance
            z: (d x n x N) observed trajectories in n trials with d dimensions and N timesteps

        Returns:
            numpyro.dist object, the conditional distribution p(x | theta)
        """
        d, n, N = z.shape

        mu, Sigma = self.conditional_moments(z, max_iter=max_iter, eps=eps)

        return dist.MultivariateNormal(mu[..., :d], Sigma[:, :, :d, :d])

    def conditional_moments(self, obs, max_iter=50, eps=1e-14):
        x = obs
        d, n, N = x.shape
        cp = self.cp
        xdim = cp.xdim

        # if controller and filter not determined yet, solve problem
        if self.solver is None:
            self.solve_problem(max_iter=max_iter, eps=eps)
        soc = self.solver

        K = soc.K
        L = soc.L

        # joint dynamics
        F1 = jnp.stack([jnp.vstack([cp.A, K[t] @ cp.H]) for t in range(N - 1)])
        F2 = jnp.stack([jnp.vstack([-cp.B @ L[t], cp.A - cp.B @ L[t] - K[t] @ cp.H]) for t in range(N - 1)])

        # joint noise covariance Cholesky factors
        G = jnp.stack([jnp.block([
            [cp.C0, jnp.zeros((cp.xdim, cp.ydim)), jnp.zeros_like(cp.A)],
            [jnp.zeros_like(cp.A), K[t] @ cp.D0, cp.E0]]) for t in range(N - 1)])

        # signal-dependent noise matrices
        M = jnp.stack([jnp.concatenate([jnp.zeros((cp.D.shape[2], xdim, xdim)),
                                        K[t] @ cp.D.transpose((2, 0, 1))], axis=1) for t in range(N - 1)])

        Mh = jnp.stack([jnp.concatenate([- cp.B @ cp.C.transpose((2, 0, 1)) @ L[t],
                                         jnp.zeros((cp.C.shape[2], xdim, xdim))], axis=1) for t in range(N - 1)])

        # intialize mean
        mu = jnp.repeat(jnp.vstack((jnp.repeat(cp.B0, 2, axis=0))), n, axis=1).T

        # initialize covariance
        Sigma = jnp.repeat(jnp.block([[cp.V0, jnp.zeros_like(cp.V0.T)],
                                      [jnp.zeros_like(cp.V0), cp.S0]])[None], n, axis=0)

        def f(carry, item):
            xt, F1, F2, G, M, Mh = item

            mu, Sigma = carry

            # conditioning: p(xhat_t | x_1:t)
            Sxh = Sigma[:, :d, d:]
            Shh = Sigma[:, d:, d:]
            Sxx = Sigma[:, :d, :d]
            Shx = Sigma[:, d:, :d]
            mu_x = mu[:, :d]
            mu_ba = mu[:, d:] + jnp.sum(Shx * jnp.linalg.solve(Sxx, xt - mu_x)[:, None, :], axis=-1)
            Sigma_ba = Shh - Shx @ jnp.linalg.solve(Sxx, Sxh)

            # updating: p(x_t+1, xhat_t+1 | x_1:t)
            mu = mu_ba @ F2.T + xt @ F1.T
            rsm_xh = Sigma_ba + vector_square(mu_ba)
            TM = jnp.sum(M[None] @ vector_square(xt)[:, None] @ M.transpose((0, 2, 1))[None], axis=1)
            TMh = jnp.sum(Mh[None] @ rsm_xh[:, None] @ Mh.transpose((0, 2, 1))[None], axis=1)
            Sigma = TM + TMh
            Sigma += quadratic_form(F2, Sigma_ba) + G @ G.T

            # for numerical stability, add small diagonal term
            Sigma += jnp.repeat(jnp.eye(Sigma.shape[-1])[None], Sigma.shape[0], axis=0) * 1e-7

            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(f, (mu, Sigma), (x.transpose((2, 1, 0))[:-1], F1, F2, G, M, Mh))

        return mu, Sigma


class PartialObservableApproximateInference(ApproximateInference):
    def conditional_distribution(self, z, max_iter=50, eps=1e-14):
        """

        Args:
            cp: control problem instance
            z: (d x n x N) observed trajectories in n trials with d dimensions and N timesteps

        Returns:
            numpyro.dist object, the conditional distribution p(x | theta)
        """
        d, n, N = z.shape

        mu, Sigma = self.conditional_moments(z, max_iter=max_iter, eps=eps)

        return dist.MultivariateNormal(mu[..., -d:], Sigma[:, :, -d:, -d:])

    def conditional_moments(self, obs, max_iter=50, eps=1e-14):
        z = obs
        d, n, N = z.shape
        cp = self.cp
        xdim = cp.xdim
        zdim = cp.zdim

        # if controller and filter not determined yet, solve problem
        if self.solver is None:
            self.solve_problem(max_iter=max_iter, eps=eps)
        soc = self.solver

        K = soc.K
        L = soc.L

        # joint dynamics
        F = jnp.stack([jnp.block([
            [cp.A, -cp.B @ L[t]],
            [K[t] @ cp.H, cp.A - cp.B @ L[t] - K[t] @ cp.H],
            [cp.S @ cp.A, -cp.S @ cp.B @ L[t]]]) for t in range(N - 1)])

        # joint noise covariance Cholesky factor
        G = jnp.stack([jnp.block([
            [cp.C0, jnp.zeros((cp.xdim, cp.ydim)), jnp.zeros_like(cp.A), jnp.zeros((cp.xdim, cp.zdim))],
            [jnp.zeros_like(cp.A), K[t] @ cp.D0, cp.E0, jnp.zeros((cp.xdim, cp.zdim))],
            [cp.S @ cp.C0, jnp.zeros((cp.zdim, cp.xdim + cp.ydim)), cp.U]]) for t in range(N - 1)])

        # signal-dependent noise matrices
        M = jnp.stack([jnp.concatenate([jnp.zeros((cp.D.shape[2], xdim, xdim)),
                                        K[t] @ cp.D.transpose((2, 0, 1)),
                                        jnp.zeros((cp.D.shape[2], zdim, xdim))], axis=1) for t in range(N - 1)])

        Mh = jnp.stack([jnp.concatenate([- cp.B @ cp.C.transpose((2, 0, 1)) @ L[t],
                                         jnp.zeros((cp.C.shape[2], xdim, xdim)),
                                         - cp.S @ cp.B @ cp.C.transpose((2, 0, 1)) @ L[t]], axis=1) for t in
                        range(N - 1)])

        # initialize mean
        mu = jnp.repeat(jnp.vstack((jnp.repeat(cp.B0, 2, axis=0), cp.S @ cp.B0)), n, axis=1).T

        # initialize covariance
        SV0 = cp.S @ cp.V0
        Sigma = jnp.repeat(jnp.block([[cp.V0, cp.V0.T, SV0.T],
                                      [cp.V0, cp.S0 + cp.V0.T, SV0.T],
                                      [SV0, SV0, cp.U @ cp.U.T + SV0 @ cp.S.T]])[None], n, axis=0)

        def f(carry, item):
            zt, F, G, M, Mh = item

            mu, Sigma = carry

            # conditioning: p(x_t, xhat_t | z_1:t)
            Sxz = Sigma[:, :-d, -d:]
            Szz = Sigma[:, -d:, -d:]
            Sxx = Sigma[:, :-d, :-d]
            Szx = Sigma[:, -d:, :-d]
            mu_z = mu[:, -d:]
            mu_ba = mu[:, :-d] + jnp.sum(Sxz * jnp.linalg.solve(Szz, zt - mu_z)[:, None, :], axis=-1)
            Sigma_ba = Sxx - Sxz @ jnp.linalg.solve(Szz, Szx)

            # updating: p(x_t+1, xhat_t+1, z_t+1 | z_t1:t)
            mu = mu_ba @ F.T
            rsm_x = Sigma_ba[:, :xdim, :xdim] + vector_square(mu_ba[:, :xdim])  # raw second moment of x
            rsm_xh = Sigma_ba[:, xdim:, xdim:] + vector_square(mu_ba[:, xdim:])  # raw second moment of x_hat
            TM = jnp.sum(M[None] @ rsm_x[:, None] @ M.transpose((0, 2, 1))[None], axis=1)  # transformed rsm of x
            TMh = jnp.sum(Mh[None] @ rsm_xh[:, None] @ Mh.transpose((0, 2, 1))[None],
                          axis=1)  # transformed rsm of x_hat
            Sigma = TM + TMh
            Sigma += quadratic_form(F, Sigma_ba) + G @ G.T

            # for numerical stability, add small diagonal term
            Sigma += jnp.repeat(jnp.eye(Sigma.shape[-1])[None], Sigma.shape[0], axis=0) * 1e-7

            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(f, (mu, Sigma), (z.transpose((2, 1, 0))[:-1], F, G, M, Mh))

        return mu, Sigma


class ApproximateInferenceFactory:

    @staticmethod
    def create(cp, solver=None):
        if isinstance(cp, PartiallyObservableControlTask):
            return PartialObservableApproximateInference(cp, solver)
        else:
            return FullyObservableApproximateInference(cp, solver)

    @staticmethod
    def get_class(cp):
        if isinstance(cp, PartiallyObservableControlTask):
            return PartialObservableApproximateInference
        else:
            return FullyObservableApproximateInference
