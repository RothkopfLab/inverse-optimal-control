import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import vmap

from ioc.examples.reaching import ReachingProblem
from ioc.methods.infer import ApproximateInferenceFactory
from ioc.methods.mle import max_likelihood
from ioc.methods.solvers import TodorovSOC

if __name__ == '__main__':
    T = 30
    r_true = -5.
    cp = ReachingProblem(T=T, r=r_true)

    # initialize Todorov solver and run
    soc = TodorovSOC(cp)
    max_iter = 50
    eps = 1e-14
    costs = soc.run(max_iter=max_iter, eps=eps)

    plt.figure()
    plt.plot(np.log10(np.abs(np.diff(costs)) + 1e-6))
    plt.show()

    num_simulations = 20
    Xa = soc.avg_trajectory()
    XSim, cost_sim = soc.simulate_trajectories(num_simulations)

    # visualize results
    plt.figure()
    plt.subplot(3, 1, 1)
    if num_simulations > 0:
        plt.plot(XSim[0].T)
    plt.plot(Xa[0], 'k', linewidth=2.)
    plt.xlabel('time step')
    plt.ylabel('position')

    plt.subplot(3, 1, 2)
    if num_simulations > 0:
        plt.plot(XSim[1].T)
    plt.plot(Xa[1], 'k', linewidth=2.)
    plt.xlabel('time step')
    plt.ylabel('velocity')

    plt.subplot(3, 1, 3)
    if num_simulations > 0:
        plt.plot(XSim[2].T)
    plt.plot(Xa[2], 'k', linewidth=2.)
    plt.xlabel('time step')
    plt.ylabel('acceleration')

    plt.show()

    # define vectorized log-likelihood function
    ll = vmap(lambda r: ApproximateInferenceFactory.create(ReachingProblem(r=r, T=T)).log_likelihood(XSim))

    # maximum likelihood estimation
    res = max_likelihood(ReachingProblem, XSim, x0=dict(r=-3., v=-1., f=-1.), method="bobyqa")

    plt.figure()

    # plot likelihood for a range of values
    rs = jnp.linspace(-7, -3)
    plt.plot(rs, ll(rs))

    # plot true and inferred
    plt.axvline(r_true, label="True", color="C1")
    plt.axvline(res.x["r"], label="MLE", color="C2")
    plt.xlabel("r")
    plt.ylabel("Log likelihood")
    plt.legend()
    plt.show()
