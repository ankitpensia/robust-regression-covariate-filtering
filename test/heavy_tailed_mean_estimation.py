import numpy as np
import matplotlib.pyplot as plt
import test.filter_algorithm as filter_algorithm
import RobustMeanEstimators
import util
import multiprocessing as mp
import pandas as pd
from tqdm.contrib.concurrent import process_map  # or thread_map
import os

# This file compares the performance of the sample mean and the robust mean estimation algorithm
# on heavy-tailed data. 
# The heavy-tailed data is generated using the pareto distribution
# with parameter 2.1.


def compare_mean_estimation(X, estimators, true_mean=None):
    num_of_methods = len(estimators)
    est_names = [est.name for est in estimators]
    error_dict = dict.fromkeys(est_names, 0.0)
    for est in estimators:
        err = est.estimateError(X, true_mean)
        error_dict[est.name] = err
    return error_dict


# Generate the data and compare the mean estimation
def sample_n_compare_mean_single(estimators, n, d, data_dist=None):
    if data_dist is None:
        X = util.sample_heavy_covariates(n, d, pareto_param=2.1)
        true_mean = None
    error_dict = compare_mean_estimation(X, estimators, true_mean)
    error_dict["n"] = n
    error_dict["d"] = d
    return error_dict

#  A wrapper function to unpack the arguments for the parallel processing
def sample_n_compare_mean_single_unpack(args):
    return sample_n_compare_mean_single(*args)


# Run the experiment in parallel
def run_experiment_parallel(num_exp, estimators, n, d, data_dist, cores=4):
    if cores is None:
        cores = mp.cpu_count() - 1
    jobs = [(estimators, n, d, data_dist)] * num_exp
    errors_list = process_map(
        sample_n_compare_mean_single_unpack, jobs, max_workers=cores,chunksize=1
    )
    # with mp.Pool(processes = cores) as pool:
    #     errors_list = pool.starmap(sample_n_compare_mean_single_unpack, jobs)
    error_pd = pd.DataFrame(errors_list)
    return error_pd



if __name__ == "__main__":
    num_exp = 500
    n = 400
    d = 100
    data_dist = None
    cores = util._DEFAULT_CORES
    sm = RobustMeanEstimators.SampleMean()
    steps = int(n * 0.05)
    pbr = RobustMeanEstimators.StochasticFilterPBR(steps=steps)
    estimators = [sm, pbr]

    filename = f"test_Heavy_tailed_Mean_num_exp_{num_exp}_n_{n}_d_{d}_est_{len(estimators)}_x_{util.rand_string(5)}.pickle"
    filename = os.path.join("data", filename)

    results = run_experiment_parallel(num_exp, estimators, n, d, data_dist,cores=cores)
    results.to_pickle(filename)

    util.plot_quantiles(results, show_max=True, y_lim=10)
