import numpy as np
import matplotlib.pyplot as plt

import util
import RobustMeanEstimators
import Regressors
import pandas as pd
import os.path


def sanity_check(n, d, estimators, ls):
    X = util.sample_heavy_covariates(n, d)
    beta_star = np.random.randn(d)
    beta_star = beta_star / np.linalg.norm(beta_star)
    eps = util.sample_heavy_response(n)
    y = X.dot(beta_star) + eps
    logs = {}
    for est in estimators:
        print(f"\nEstimator: {est.name}")
        beta_hat, logs_est = est.estimateBeta(
            X, y, verbose=True, true_beta=beta_star, logs_=True
        )
        logs[est.name] = logs_est
    util.plot_error_two(logs, title="SanityCheck")
    print(f"\nLeast squares error: {ls.estimateError(X,y,beta_star)}")


if __name__ == "__main__":
    n = 200
    d = 20

    beta_0 = np.random.randn(d)
    beta_0 = (beta_0 / np.linalg.norm(beta_0)) * 1
   
    data_dist_dict = {
        "name": "heavy",
        "pareto_param_x": 2,
        "pareto_param_y": 2,
    }

    ls = Regressors.LeastSquares()

    estimators = []
    for i in range(-3, 1):
        for j in range(0, 3):
            thres = 2**i
            filter_steps = int(n * j * 0.05)
            pbr = RobustMeanEstimators.StochasticFilterPBR(steps=filter_steps)
            bt_hub = Regressors.Filter_HuberRegression_GD_LineSearch_EarlyStopping(
                thres=thres, stopping=5e-4, covFilter=pbr
            )
            estimators.append(bt_hub)

    sanity_check(n, d, estimators, ls)

    estimators.append(ls)

    num_exp = int(5e2)

    filename = f"test_Hub_LS_effect_of_threshold_Filter_Reg_num_exp_{num_exp}_n_{n}_d_{d}_est_{len(estimators)}_x_{util.rand_string(5)}.pickle"
    filename = os.path.join("data", filename)
    results = util.run_experiment_serial(num_exp, estimators, n, d, data_dist_dict)
    results.to_pickle(filename)
    util.plot_quantiles(results, show_max=True)
