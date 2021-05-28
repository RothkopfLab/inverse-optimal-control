import matplotlib.pyplot as plt
from jax import jit, vmap
import jax.numpy as jnp

from ioc.methods.infer import ApproximateInferenceFactory
from ioc.methods.solvers import TodorovSOC
from ioc.methods.mle import max_likelihood
from ioc.examples import create_random_problem

# generate a new random problem class
seed = 1
RandomProblem = create_random_problem(seed=seed)

# initialize with true parameters
r1_true = -2
r2_true = -5
ex = RandomProblem(r1=r1_true, r2=r2_true, T=50)

# run solver
max_iter = 50
eps = 0
lqg = TodorovSOC(ex)
cost = lqg.run(max_iter=max_iter, eps=eps)

# simulate data
XSim, _ = lqg.simulate_trajectories(50)

# get likelihood function
cls = ApproximateInferenceFactory.get_class(RandomProblem)
ll = jit(vmap(
    lambda r: cls(RandomProblem(r1=r[0], r2=r[1])).log_likelihood(XSim, max_iter=max_iter, eps=eps)))

# compute log likelihood on a grid
n = 30
r_lo, r_hi = -7, 1
r = jnp.linspace(r_lo, r_hi, n)
rs = jnp.array(jnp.meshgrid(r, r)).T.reshape(-1, 2)
nll = -ll(rs).reshape(n, n).T
# normalize for visualization
nll = (nll - nll.min()) / (nll.max() - nll.min())

# plot (negative) log likelihood
plt.imshow(nll, extent=[r_lo, r_hi, r_hi, r_lo], vmax=0.1)
plt.ylabel("r2")
plt.xlabel("r1")
plt.scatter(r1_true, r2_true, marker="x", color="red", label="true")

# compute maximum likelihood estimate
res = max_likelihood(RandomProblem, XSim, x0=None, bounds=dict(r1=(r_lo, r_hi), r2=(r_lo, r_hi)), method="bobyqa",
                     ntimes=5)
mle = res.x

plt.scatter(mle["r1"], mle["r2"], marker="o", color="magenta", label="MLE")
plt.title(f"Random problem, seed {seed}")
plt.legend()
plt.show()
