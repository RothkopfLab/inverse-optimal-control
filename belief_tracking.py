import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, vmap
import matplotlib as mpl

from ioc.methods.infer import ApproximateInferenceFactory
from ioc.methods.solvers import TodorovSOC
from ioc.examples import PartiallyObservedReachingProblem


def get_handles_labels(fig, **kwdargs):
    # generate a sequence of tuples, each contains
    #  - a list of handles (lohand) and
    #  - a list of labels (lolbl)
    tuples_lohand_lolbl = (ax.get_legend_handles_labels() for ax in fig.axes)
    # e.g. a figure with two axes, ax0 with two curves, ax1 with one curve
    # yields:   ([ax0h0, ax0h1], [ax0l0, ax0l1]) and ([ax1h0], [ax1l0])

    # legend needs a list of handles and a list of labels,
    # so our first step is to transpose our data,
    # generating two tuples of lists of homogeneous stuff(tolohs), i.e
    # we yield ([ax0h0, ax0h1], [ax1h0]) and ([ax0l0, ax0l1], [ax1l0])
    tolohs = zip(*tuples_lohand_lolbl)

    # finally we need to concatenate the individual lists in the two
    # lists of lists: [ax0h0, ax0h1, ax1h0] and [ax0l0, ax0l1, ax1l0]
    # a possible solution is to sum the sublists - we use unpacking
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)

    return handles, labels


quad = vmap(lambda x, Q: x.T @ Q @ x, in_axes=(0, None))

if __name__ == '__main__':
    model = PartiallyObservedReachingProblem(obs_noise=5e-3)

    soc = TodorovSOC(model)
    soc.run()

    np.random.seed(0)
    XObs, cost_sim, XSim, Xhat, Us = soc.simulate_trajectories(20, return_x_sim=True)

    inf = ApproximateInferenceFactory.create(model, soc)

    mu, Sigma = inf.conditional_moments(XObs)

    trial = 2
    t = jnp.arange(29) / 100
    f, ax = plt.subplots(3, 1, sharex=True)

    mu_pos = mu[:, trial, 5]
    std_pos = jnp.sqrt(Sigma[:, trial, 5, 5])
    ax[0].plot(t, XObs[0, trial, 1:], label="experimenter's observation")  # plot trajectory
    ax[0].plot(t, Xhat[0, trial, 1:], color="C2", label="agent's belief", linestyle=":")
    ax[0].plot(t, mu_pos, linestyle="--", color="C1", label="estimate of belief")
    ax[0].fill_between(t, mu_pos - std_pos * 2, mu_pos + std_pos * 2,
                       alpha=0.5, color="C1", label="experimenter's uncertainty")
    ax[0].set_yticks([0., .1])
    ax[0].set_title("position", fontsize=6)

    # ax[0].legend(fontsize=6)
    # ax[0].set_title("Belief tracking")

    ax[1].plot(t, XSim[1, trial, 1:], color="C3", label="unobserved vel. / acc.")

    mu_vel = mu[:, trial, 6]
    std_vel = jnp.sqrt(Sigma[:, trial, 6, 6])
    ax[1].plot(t, Xhat[1, trial, 1:], color="C2", linestyle=":")
    ax[1].plot(t, mu_vel, color="C1", linestyle="--")  # plot our belief about acceleration
    ax[1].fill_between(t, mu_vel - std_vel * 2, mu_vel + std_vel * 2, alpha=0.5, color="C1")
    # ax[1].legend(fontsize=6)
    ax[1].set_title("velocity", fontsize=6)

    ax[2].plot(t, XSim[2, trial, 1:], color="C3")

    mu_acc = mu[:, trial, 7]
    std_acc = jnp.sqrt(Sigma[:, trial, 7, 7])
    ax[2].plot(t, Xhat[2, trial, 1:], color="C2", linestyle=":")
    ax[2].plot(t, mu_acc, color="C1", linestyle="--")  # plot our belief about acceleration
    ax[2].fill_between(t, mu_acc - std_acc * 2, mu_acc + std_acc * 2, alpha=0.5, color="C1")
    # ax[2].legend(fontsize=6)
    ax[2].set_title("acceleration", fontsize=6)

    handles, labels = get_handles_labels(f)
    handles = [handles[0], handles[1], handles[4], handles[2], handles[3]]
    labels = [labels[0], labels[1], labels[4], labels[2], labels[3]]

    kwargs = dict(fontsize=4, bbox_to_anchor=[0.5, -0.1],
                  loc='center', ncol=2)
    f.legend(handles, labels, **kwargs)

    f.tight_layout()
    plt.show()
