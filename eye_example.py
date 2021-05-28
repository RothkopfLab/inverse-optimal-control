from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ioc.methods.infer import ApproximateInferenceFactory
from ioc.methods.mle import max_likelihood
from ioc.methods.solvers import TodorovSOC, KalmanInit
from ioc.examples import SaccadeModel

r_true = -4.5
dt = 1.25
T = 40
ti = 5
cp = SaccadeModel(r=r_true, T=40, ti=ti)

# initialize Todorov solver and run
soc = TodorovSOC(cp)
max_iter = 50
eps = 0.

np.random.seed(1)
costs = soc.run(max_iter=max_iter, eps=eps, init=KalmanInit.RANDOM)
np.random.seed(0)

num_simulations = 20
Xa = soc.avg_trajectory()
XObs, cost_sim, XSim, Xhat, Us = soc.simulate_trajectories(num_simulations, return_x_sim=True)

# visualize results

fig, ax = plt.subplots(2, figsize=(8, 8), sharex=True)
t = np.linspace(0, 45, T) * dt

ax[0].axvline(ti * dt, linestyle="--", color="gray", linewidth=.5)
ax[0].axvline(35 * dt, linestyle="--", color="gray", linewidth=.5)
ax[0].axhline(-10, color="red", linewidth=.5)
ax[0].axhline(10, color="red", linewidth=.5)

ax[0].plot(t, XSim[0].T)

ax[0].set_xticks([ti * dt, 52.5])
ax[0].set_xticklabels(["0", "40"])

ax[0].set_ylabel("Angle")

ax[1].plot(t[:-1], Us[0].T)

ax[1].set_xlabel("time [ms]")
ax[1].set_ylabel("Control")
plt.tight_layout()
plt.show()

opt_range = {
    'r': [-7., 0.],
}

mle = max_likelihood(SaccadeModel, XObs, x0=None, bounds=opt_range, method="bobyqa", max_iter=max_iter,
                     eps=eps, ntimes=5)
sol = mle.x

ll = lambda r: ApproximateInferenceFactory.create(SaccadeModel(r=r)).log_likelihood(
    jnp.array(XObs),
    max_iter=max_iter,
    eps=eps)

# rs = jnp.linspace(-7, 0.)
rs = jnp.linspace(-6., -2.)
lls = vmap(ll)(rs)

plt.figure()
plt.plot(rs, lls)
plt.axvline(r_true, label="True", color="C1")
plt.axvline(sol["r"], label="MLE", color="C2")
plt.xlabel("r")
plt.ylabel("Log likelihood")
plt.legend()
plt.show()
