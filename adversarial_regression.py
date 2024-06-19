import numpy as np
import matplotlib.pyplot as plt

import util
import RobustMeanEstimators
import Regressors
from tqdm.contrib.concurrent import process_map  # or thread_map
import pandas as pd
import os.path
from tqdm import tqdm



def sanity_check_adv(
    n, d, outliers_frac, estimators, ls, x_scale=10, y_x_ratio=100, pareto_param=4
):
    np.random.seed()

    X = util.sample_heavy_covariates(n, d, pareto_param=pareto_param)
    eps = util.sample_heavy_response(n)

    outliers_1 = int(n * outliers_frac * 0.5)
    outliers_2 = int(n * outliers_frac * 0.5)

    X[n - outliers_1 - outliers_2 : n - n - outliers_1, :] = np.ones((outliers_1, d)) * (x_scale)
    beta_star = np.random.randn(d)
    beta_star = beta_star / np.linalg.norm(beta_star)
    y = X.dot(beta_star) + eps
    y[n - outliers_1 - outliers_2 :] = y_x_ratio * x_scale
    logs = {}
    for est in estimators:
        print(f"\nEstimator: {est.name}")
        beta_hat, logs_est = est.estimateBeta(
            X, y, verbose=True, true_beta=beta_star, logs_=True
        )
        logs[est.name] = logs_est
        print(f": {logs_est[-1,1]:0.5f}")

    util.plot_error_two(logs, title="SanityCheck")
    print(f"\nLeast squares error: {ls.estimateError(X,y,beta_star)}")


if __name__ == "__main__":
    cores = util._DEFAULT_CORES
    n = 200
    d = 40
    outliers_frac = 0.1
    beta_0 = np.random.randn(d)
    beta_0 = (beta_0 / np.linalg.norm(beta_0)) * 1
   
    num_exp = int(5e4)
    x_scale = 10
    y_x_ratio = 20
    pareto_param_x = 4
    pareto_param_y = 2

    data_dist_dict = {
        "name": "outlliers_reg",
        "pareto_param_x": pareto_param_x,
        "pareto_param_y": pareto_param_y,
        "x_scale": x_scale,
        "y_x_ratio": y_x_ratio,
        "outliers_frac": outliers_frac,
    }


    thres, stopping, min_, max_, maxiters = 0.5, 5e-4, -8, 2, 1e2
    hard_thres_size = int(n * outliers_frac * 1.5)
    lts_steps = 100

    hb = Regressors.HuberRegression_GD_LineSearch_EarlyStopping(
        thres, stopping, beta_0, min_=min_, max_=max_, max_iters_=maxiters
    )
    ls = Regressors.LeastSquares()

    filter_steps = int(n * outliers_frac * 1.5)
    pbr = RobustMeanEstimators.StochasticFilterPBR(steps=filter_steps)
    filtered_hb = Regressors.Filter_HuberRegression_GD_LineSearch_EarlyStopping(
        thres,
        stopping,
        beta_0,
        min_=min_,
        max_=max_,
        max_iters_=maxiters,
        covFilter=pbr,
    )

    lts = Regressors.LTS_Alt_Min(
        steps=lts_steps, hard_thres_size=hard_thres_size
    )

    filtered_lts = Regressors.Filter_LTS_Alt_Min(
        steps=lts_steps, hard_thres_size=hard_thres_size, covFilter=pbr
    )



    sanity_check_adv(
        n,
        d,
        outliers_frac,
        [hb, filtered_hb, lts, filtered_lts],
        ls,
        x_scale,
        y_x_ratio,
        pareto_param=pareto_param_x,
    )


    estimators = [hb, ls, filtered_hb, lts, filtered_lts]

    filename = f"Adv_both_xy_x_scale_{x_scale}_y_x_ratio_{y_x_ratio}_Reg_num_exp_{num_exp}_n_{n}_d_{d}_eps_{outliers_frac}_est_{len(estimators)}_pareto_{pareto_param_x}_var_{pareto_param_y}_x_{util.rand_string(5)}.pickle"
    filename = os.path.join("data", filename)
    print(filename)
    results = util.run_experiment_parallel(
        num_exp,
        estimators,
        n,
        d,
        data_dist_dict,
        cores=cores)
    
    results.to_pickle(filename)

    util.plot_quantiles(results, show_max=True, y_lim=5)
