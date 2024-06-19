import numpy as np
import matplotlib.pyplot as plt

import util
import RobustMeanEstimators
import Regressors
from tqdm.contrib.concurrent import process_map  # or thread_map
import pandas as pd
import os.path
from tqdm import tqdm


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
    cores = util._DEFAULT_CORES
    n = 200
    d = 40
    num_exp = int(5e4)

    pareto_param_x = 2
    pareto_param_y = 2
    data_dist_dict = {
        "name": "heavy",
        "pareto_param_x": pareto_param_x,
        "pareto_param_y": pareto_param_y,
    }

    

    beta_0 = np.random.randn(d)
    beta_0 = (beta_0 / np.linalg.norm(beta_0)) * 1

    ls = Regressors.LeastSquares()
    ransac = Regressors.RANSACRegression()
    theilsen = Regressors.TheilSenRegression()

    thres, stopping, min_, max_, maxiters = 1, 5e-4, -6, 2, 1e2
    lts_steps = 100

    estimators = []
    for i in range(-3, 2):
        for j in range(0, 2):
            thres = 2**i
            filter_steps = int(n * j * 0.05)
            pbr = RobustMeanEstimators.StochasticFilterPBR(steps=filter_steps)
            # filtered_hb = Regressors.FilterHuberGD_LS_Stopping(thres=thres,stopping=5e-4,covFilter=pbr)
            filtered_hb = Regressors.Filter_HuberRegression_GD_LineSearch_EarlyStopping(
                thres,
                stopping,
                beta_0,
                min_=min_,
                max_=max_,
                max_iters_=maxiters,
                covFilter=pbr,
            )
            estimators.append(filtered_hb)

    for i in range(1,5):
        for j in range(2):
            hard_thres_size = int(n * i * 0.05)
            filter_steps = int(n * j * 0.05)
            pbr = RobustMeanEstimators.StochasticFilterPBR(steps=filter_steps)
            lts = Regressors.Filter_LTS_Alt_Min(
                steps=lts_steps, hard_thres_size=hard_thres_size, covFilter=pbr
            )

            estimators.append(lts)

    sanity_check(n, d, estimators, ls)

    estimators.append(ls)
    estimators.append(ransac)
    estimators.append(theilsen)


    filename = f"Heavy_tailed_reg_LTS_Huber_num_exp_{num_exp}_n_{n}_d_{d}_est_{len(estimators)}__pareto_{pareto_param_x}_var_{pareto_param_y}_x_{util.rand_string(5)}.pickle"
    filename = os.path.join("data", filename)

    print(filename)
    results = util.run_experiment_parallel(
        num_exp, 
        estimators, 
        n,
        d, 
        data_dist_dict,
        cores=cores
    )
    results.to_pickle(filename)
    util.plot_quantiles(results, show_max=True, y_lim=1.5, y_min = 0.2)