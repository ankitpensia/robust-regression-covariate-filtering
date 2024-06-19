In this file, we explain the contents of this folder.
The files in this folder test individual components of the program and are used to verify that the program works as expected.


# Files in the Folder
The files in this folder are:
- `filter_algorithm.py`
    + This file tests the filter algorithm and plots how the eigenvalues of the covariance matrix and estimation error changes across the filtering steps.
- `heavy_tailed_mean_estimation.py`
    + This folder contains files that is analogous to `heavy_tailed_regression.py` in the main folder, but for the purpose of the mean estimation. Since it is not the focus of the project, the evaluation is only preliminary and compares the filtering algorithm with the sample mean.
- `huber_gd.py`
    + This file test the performance/convergence of `HuberRegressionGD` from `Regressors.py`. Since the modified version of gradient descent (with early stopping and line search) outperforms this vanilla version, we do not pursue `HuberRegressionGD` in the main experiments.
- `huber_line_search_early_stopping.py`
    + This file test the performance/convergence of `Filter_HuberRegression_GD_LineSearch_EarlyStopping` (and thus also `HuberRegression_GD_LineSearch_EarlyStopping`) from `Regressors.py`.
- `lts_alt_min_early_stopping.py`
    + This file test the performance/convergence of `Filter_LTS_Alt_Min_Early_Stopping` (and thus also `LTS_Alt_Min_Early_Stopping`) from `Regressors.py`.
- `lts_alt_min.py`
    + This file test the performance/convergence of `Filter_LTS_Alt_Min` (and thus also `LTS_Alt_Min`) from `Regressors.py`. Since the modified version  (with early stopping) outperforms this vanilla version, we do not pursue `LTS_Alt_Min` in the main experiments.

# How to Run These Files

These files should be run in terminal/bash from the parent directory (and not the directory `test`).
The instructions are as follows:
1. Open terminal/bash.
2. Navigate to the parent directory of the project.
3. To run file `test/filename.py`, run `python -m test.filename`.
    + For example, to run `filter_algorithm.py`, run `python -m test.filter_algorithm`.
    + Observe that there is no `.py` and we replace `/` with a period `.`
4. The results of the simulation will be saved in the `data` folder, and a summary plot will be shown on the screen.