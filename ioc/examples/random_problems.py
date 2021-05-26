import numpy as np
from jax import random, numpy as jnp
from jax.scipy import linalg as jlinalg
from numpyro import distributions as dist

from ioc.examples.problem import ControlTask

normalize = lambda M, k: k * M / jnp.linalg.norm(M, ord=2)


def create_random_problem(seed=0, xdim=5, ydim=4):
    key = random.PRNGKey(seed)

    udim = 2

    # dynamics model
    key, subkey = random.split(key)
    # random matrix, but with last element fixed at 1.
    A = random.normal(subkey, (xdim - 1, xdim - 1))
    A = normalize(A, 1.)
    A = jlinalg.block_diag(A, jnp.array(1.))
    key, subkey = random.split(key)
    # random control matrix, last row fixed at 0.
    B = random.normal(subkey, (xdim - 1, udim))
    B = normalize(B, 1.0)
    B = jnp.vstack([B, jnp.zeros((1, udim))])

    # observation model
    key, subkey = random.split(key)
    H = random.normal(subkey, (ydim, xdim - 1))
    H = jnp.eye(ydim, xdim - 1)
    H = jnp.hstack([H, jnp.zeros((ydim, 1))])

    # noise model
    key, subkey = random.split(key)
    C0 = dist.LKJCholesky(xdim - 1, concentration=10.).sample(subkey) * .5
    C0 = jlinalg.block_diag(C0, jnp.array(0.))

    # control-dependent noise
    key, subkey = random.split(key)
    C = random.uniform(subkey, (udim, udim, 1), maxval=.5)  # [..., np.newaxis]

    # observation noise model (only fixed small noise for target dimension)
    key, subkey = random.split(key)
    D0 = dist.LKJCholesky(ydim, concentration=10.).sample(subkey) * .5

    # state-dependent observation noise (not for target dimension)
    key, subkey = random.split(key)
    D = random.uniform(subkey, (ydim, xdim - 1, 1), maxval=.5)
    D = jnp.hstack([D, jnp.zeros((ydim, 1, 1))])

    # estimation noise
    key, subkey = random.split(key)
    E0 = dist.LKJCholesky(xdim - 1, concentration=10.).sample(subkey) * .5
    E0 = jlinalg.block_diag(E0, jnp.array(0.))

    # initial values: completely zero, but last element fixed at 1.
    x0 = jnp.vstack([jnp.zeros((A.shape[0] - 1, 1)), jnp.array([[1.]])])
    S0 = jnp.zeros_like(A)

    # cost function
    # goal is to bring the first element of the state to the target
    d = np.zeros((3, xdim))
    d[0, 0] = 1
    d[0, xdim - 1] = -1

    # define the constructor for our new class
    def constructor(self, r1, r2, T=50):
        R = jnp.diag(jnp.array([10 ** r1, 10 ** r2]))

        Q = np.zeros((xdim, xdim, T))
        Q[..., -1] = d.T @ d
        Q = jnp.array(Q)

        ControlTask.__init__(self, A, B, C, C0, H, D, D0, E0, Q, R, x0, S0)

    # create a new type with our pre-defined random problem
    return type("RandomProblem", (ControlTask,), {
        "__init__": constructor
    })
