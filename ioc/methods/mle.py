import inspect

from numpy.random import default_rng

from ioc.methods.infer import ApproximateInferenceFactory
from ioc.methods.optimize import minimize, minimize_bobyqa
import numpy as np
import jax


def max_likelihood(model_type, XSim, x0, bounds=None, method="Nelder-Mead",
                   max_iter=50, eps=1e-14, ntimes=1, **fixed_params):
    d, n, T = XSim.shape
    random_x0 = x0 is None

    params = {}
    for name, default in get_model_params(model_type).items():
        if name not in fixed_params:
            params[name] = default

    def nll(params):
        cp = model_type(**params, **fixed_params, T=T)
        return -ApproximateInferenceFactory.create(cp).log_likelihood(XSim, max_iter=max_iter, eps=eps)

    nll = jax.jit(nll)

    ress = []
    for i in range(ntimes):
        if random_x0:
            x0 = {}
            for k, v in bounds.items():
                x0[k] = np.random.uniform(*v)

        # find the maximum of the likelihood function (in log space)
        if method in ["Nelder-Mead", "L-BFGS-B"]:
            res = minimize(nll, x0=x0, method=method, bounds=bounds)
        elif method == "bobyqa":
            res = minimize_bobyqa(nll, x0=x0, bounds=bounds)
        else:
            raise ValueError("Please choose a valid optimization method")

        ress.append(res)

    min_idx = min(enumerate(ress), key=lambda x: x[1].f)[0]

    return ress[min_idx]


def get_model_params(model_class):
    init_signature = inspect.signature(model_class.__init__)

    parameters = {}
    for name, param in init_signature.parameters.items():
        if name not in ["self", "T"]:
            parameters[param.name] = param.default

    return parameters
