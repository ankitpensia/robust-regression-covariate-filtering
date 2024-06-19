# Requirements

The code is written in Python 3 and relies on the following libraries:
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [tqdm](https://pypi.org/project/tqdm/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [ipykernel](https://ipykernel.readthedocs.io/en/stable/)

These could be installed using pip:
```bash
pip install -r requirements.txt
```

# Generating Data For Plots

 

At a high level, there are three steps to generate the data for the plots. They are:
1. (Python script) Generate the data for heavy-tailed linear regression.
2. (Python script) Generate the data for adversarial linear regression.
3. (Jupyter Notebook) Finally, create the plot.

The detailed steps are given in the following subsections:

##  Step 1: Heavy-Tailed Regression

+ The file is `heavy_tailed_regression.py`.
+ Set the number of repetitions to generate the confidence intervals in `heavy_tailed_regression.py` by changing the value of `num_exp` (currently set to `50000`).
+ Set the number of cores to be used in the variable `cores`. The default value of `cores` is `_DEFAULT_CORES`  from `util.py`. 
+ The choice of data distribution is set there, where we set Pareto parameter for the covariates. The default is set to be 2.
+ Choose the list of estimators to be evaluated in the variable `estimators`. Currently, Huber regression, LTS regression, OLS, Theil-Sen, and RANSAC regression are included. The parameters for these estimators can be set in `heavy_tailed_regression.py`.
+ Run the file `heavy_tailed_regression.py` using the command `python heavy_tailed_regression.py` to generate the data for the plots. The data will be saved in the folder `data/`. The file name is set to be `Heavy_tailed_reg_LTS_Huber_[......].pickle`.

## Step 2: Adversarial Regression

The corresponding file is `adversarial_regression.py`. The instructions are the same as above, and this file could be run by executing the command `python adversarial_regression.py`, which stores the data in the folder `data/` with the filename `Adv_[...].pickle`.

## Step 3: Creating the Plot

+ Open the file `plots_for_paper.ipybn` in Jupyter notebook.
+ Replace the filenames in the variables `heavy_tailed_results_filename` and `adversarial_results_filename` with the filenames generated in the previous steps, respectively, in the first cell of the notebook.
+ Run the remaining cells in the Jupyter notebook to generate the plots.
+ The plots are saved in the folder `fig/` with the names `heavy_[..].pdf` and `adv_[...].pdf`.



**Important**: If you are generating new data (instead of using the existing pickle files), please note that the resulting figures will have some random fluctuations. (If you are using the existing pickle files---by directly going to Step 3--- then the figures will be exact).


# Important Files

The following files contain the implementations:

1. `util.py`: This file contains helper functions pertaining to linear algebra, random sampling, and plotting.

2. `RobustMeanEstimators.py`:  This file implements the filtering algorithm for robust mean estimation. These filtering algorithms are used to remove samples whose covariates are deemed to be outliers by the filtering algorithm.

3. `Regressors.py`: This file implements different kinds of estimators for linear regression. Some of these are the vanilla version of the estimators that act as baseline: OLS, RANSAC, Theil-Sen, Huber Regression, Least Trimmed Squares. Importantly, we also define the versions of these regressors that perform a simple pre-processing of covariate filtering.


# Auxiliary Files

The files in the folder `test` evaluates the performance of various estimators (in isolation) to ensure that these estimators are converging with the chosen value of hyperparameters. Please see `tests/readme.md` for further details.

