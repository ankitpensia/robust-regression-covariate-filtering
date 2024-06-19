# This file implements different kinds of estimators for linear regression.
import numpy as np
from scipy.linalg import eigh
from abc import ABC, abstractmethod
import tqdm
import scipy.special
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor


"""
This file implements different kinds of estimators for linear regression. 
In particular, they include the following:

--------Baselines--------
1. Least Squares
2. RANSAC
3. Theil-Sen
4. Huber Regression with Gradient Descent
5. Huber Regression with Modified Gradient Descent
    Line Search and Early Stopping Criteria
6. Least Trimmed Squares with Alternating Minimization
7. Least Trimmed Squares with Modified Alternating Minimization
    Early Stopping Criterion

------------Covariate Filtering-----------

8. Filtered Huber Regression with Gradient Descent
9. Filtered Huber Regression with Modified Gradient Descent
10. Filtered Least Trimmed Squares with Alternating Minimization
11. Filtered Least Trimmed Squares with Modified Alternating Minimization

In the experiments, we use the modified version of Huber Regression (5 and 9 above) 
and Least Trimmed Squares with Alternating Minimization (6 and 10 above).
"""


# -----------------------------------------------------------
# Helper functions for linear regression and Huber regression
def leastsquares(X, y):
    """
    Compute the least squares solution for a linear regression problem.

    Parameters:
    X (array-like): The input feature matrix of shape (n_samples, n_features).
    y (array-like): The target values of shape (n_samples,).

    Returns:
    b_ols (array-like): The estimated coefficients of the linear regression model.

    """
    b_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=0.01)
    return b_ols


def huber(x, thres):
    # Huber loss function
    return scipy.special.huber(thres, x)


def get_residual(beta, X, y):
    # Compute the residuals for the linear regression model
    return y - X.dot(beta)


def grad_huber(x, thres):
    # Compute the gradient of the Huber loss function. 
    # Input is a vector x and a threshold thres.
    # The gradient is computed element-wise.
    grad = np.zeros_like(x)
    sq_indices = np.absolute(x) <= thres
    lin_indices = np.logical_not(sq_indices)
    grad[sq_indices] = x[sq_indices]
    grad[lin_indices] = thres * np.sign(x[lin_indices])
    return grad


def grad_huber_LR(beta, X, y, thres):
    # Compute the gradient for the Huber regression model using chain rule.
    res = get_residual(beta, X, y)
    grad_res = grad_huber(res, thres)
    grad = -X.T.dot(grad_res) / X.shape[0]
    return grad


def huber_LR(beta, X, y, thres):
    # Compute the (average) Huber loss function for the linear regression model.
    res = get_residual(beta, X, y)
    return huber(res, thres).mean()


def hard_thresholding(val, m):
    # Hard thresholding operator
    small_indices = np.absolute(val).argsort()[:-m]
    val[small_indices] = 0
    return val


def hat_matrix(X):
    # Compute the hat matrix for the linear regression model
    return X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)


def lts_error(X, y, beta, m):
    # Compute the (average) least trimmed squares error for the linear regression model 
    res = get_residual(beta, X, y)
    res_sq = np.power(res, 2)
    res_sq_srt = np.sort(res_sq)
    return res_sq_srt[:-m].mean()

# ------------------------------------------------
# An abstract wrapper class for linear regression models.
class LinearRegressor(ABC):
    """
    Abstract base class for linear regression models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def estimateBeta(self, X):
        """
        Estimates the beta coefficients of the linear regression model.

        Parameters:
        - X: The input features matrix.

        Returns:
        - The estimated beta coefficients.
        """
        pass

    @abstractmethod
    def estimateError(self, X, y, true_beta):
        """
        Estimates the error of the linear regression model.

        Parameters:
        - X: The input features matrix.
        - y: The target variable vector.
        - true_beta: The true beta coefficients.

        Returns:
        - The estimated error.
        """
        pass
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------

# Regressor 1: Least Squares (Baseline)
class LeastSquares(LinearRegressor):
    """
    A class representing the Least Squares linear regression algorithm.

    Attributes:
        name (str): The name of the regressor.

    Methods:
        estimateBeta(X, y, true_beta=None): Estimates the beta coefficients using the Least Squares method.
        estimateError(X, y, true_beta): Estimates the error between the predicted beta coefficients and the true beta coefficients.
    """

    def __init__(self):
        self.name = "LeastSquares"

    def estimateBeta(self, X, y, true_beta=None):
        """
        Estimates the beta coefficients using the Least Squares method.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            true_beta (array-like, optional): The true beta coefficients (default: None).

        Returns:
            array-like: The estimated beta coefficients.
        """
        return leastsquares(X, y)

    def estimateError(self, X, y, true_beta):
        """
        Estimates the error between the predicted beta coefficients and the true beta coefficients.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            true_beta (array-like): The true beta coefficients.

        Returns:
            float: The error between the predicted beta coefficients and the true beta coefficients.
        """
        beta_hat = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)

# Regressor 2: RANSAC (Baseline)
class RANSACRegression(LinearRegressor):
    """
    RANSACRegression is a class that performs robust regression using the RANSAC algorithm.

    Attributes:
        name (str): The name of the regression model.

    Methods:
        estimateBeta(X, y, true_beta=None): Estimates the regression coefficients using the RANSAC algorithm.
        estimateError(X, y, true_beta): Estimates the error between the estimated regression coefficients and the true coefficients.
    """

    def __init__(self):
        self.name = "Ransac"

    def estimateBeta(self, X, y, true_beta=None):
        """
        Estimates the regression coefficients using the RANSAC algorithm.

        Args:
            X (array-like): The input features.
            y (array-like): The target values.
            true_beta (array-like, optional): The true regression coefficients. Defaults to None.

        Returns:
            array-like: The estimated regression coefficients.
        """
        ransac = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=False),
            random_state=42,
            max_trials=200,
        )
        ransac.fit(X, y)
        return ransac.estimator_.coef_

    def estimateError(self, X, y, true_beta):
        """
        Estimates the error between the estimated regression coefficients and the true coefficients.

        Args:
            X (array-like): The input features.
            y (array-like): The target values.
            true_beta (array-like): The true regression coefficients.

        Returns:
            float: The error between the estimated regression coefficients and the true coefficients.
        """
        beta_hat = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)

# Regressor 3: TheilSen (Baseline)
class TheilSenRegression(LinearRegressor):
    """
    TheilSenRegression is a class that implements the Theil-Sen regression algorithm.

    Attributes:
        name (str): The name of the regression algorithm.

    Methods:
        estimateBeta(X, y, true_beta=None): Estimates the coefficients (beta) of the linear regression model.
        estimateError(X, y, true_beta): Estimates the error between the predicted coefficients and the true coefficients.

    """

    def __init__(self, random_state=42,max_subpopulation=200 ):
        self.name = "TheilSen"
        self.random_state=random_state
        self.max_subpopulation = max_subpopulation

    def estimateBeta(self, X, y, true_beta=None):
        """
        Estimates the coefficients (beta) of the linear regression model.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            true_beta (array-like, optional): The true coefficients (for error estimation).

        Returns:
            array-like: The estimated coefficients.

        """
        theilsen = TheilSenRegressor(
            fit_intercept=False, random_state=self.random_state, max_subpopulation=self.max_subpopulation)
        theilsen.fit(X, y)
        return theilsen.coef_

    def estimateError(self, X, y, true_beta):
        """
        Estimates the error between the predicted coefficients and the true coefficients.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            true_beta (array-like): The true coefficients.

        Returns:
            float: The error between the predicted coefficients and the true coefficients.

        """
        beta_hat = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)


# Regressor 4: Huber Regression via (Vanilla) Gradient Descent
class HuberRegressionGD(LinearRegressor):
    def __init__(self, thres=1, iters=None, step_size=None, beta_0=None):
        self.name = f"Hub_thres_{thres}_iters_{iters}"
        self.thres = thres
        self.iters = iters
        self.step_size = step_size
        self.beta_0 = beta_0

    def estimateBeta(self, X, y, verbose=False, logs_=False, true_beta=None):
        if self.beta_0 is None:
            n, d = X.shape
            beta_0 = np.random.randn(d)
            beta_0 = beta_0 / norm(beta_0)
            beta = np.zeros_like(beta_0) + beta_0
        else:
            beta = np.zeros_like(self.beta_0) + self.beta_0

        if logs_ is True:
            logs = np.zeros((self.iters, 2))
        else:
            logs = None
        pbar = tqdm.trange(self.iters, disable=not verbose)
        for i in pbar:
            if logs_ is True:
                huber_val = huber_LR(beta, X, y, self.thres)
                l2_err = norm(beta - true_beta)
                logs[i, 0] = huber_val
                logs[i, 1] = l2_err
            beta = beta - self.step_size * grad_huber_LR(beta, X, y, self.thres)
        pbar.close()
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y, true_beta=true_beta)
        return norm(beta_hat - true_beta)

# Regressor 5: Huber Regression via Modified Gradient Descent (Baseline)
# The gradient descent uses Line Search and Early Stopping Cretrion
# The early stopping criterion is based on the norm of the gradient
# The line search is based on choosing the step size that best minimizes the Huber Loss function
# Empirically, these two techniques are used to improve the convergence of the Huber Regression model 
class HuberRegression_GD_LineSearch_EarlyStopping(LinearRegressor):
    def __init__(
        self,
        thres=1,
        stopping=None,
        beta_0=None,
        min_=-5,
        max_=5,
        range_step_=1,
        max_iters_=1e2,
    ):
        self.name = f"Hub_LS_thres_{thres}_stopping_{stopping}"
        self.thres = thres
        self.stopping = stopping
        self.beta_0 = beta_0
        self.min_ = min_
        self.max_ = max_
        self.range_step_ = range_step_
        self.max_iters_ = max_iters_
        # self.func_diff = 1e-10

    def estimateBeta(self, X, y, verbose=False, logs_=False, true_beta=None):
        if self.beta_0 is None:
            n, d = X.shape
            beta_0 = np.random.randn(d)
            beta_0 = beta_0 / norm(beta_0)
            beta = np.zeros_like(beta_0) + beta_0
        else:
            beta = np.zeros_like(self.beta_0) + self.beta_0

        if logs_ is True:
            logs = []
        else:
            logs = None
        trials = 0
        step_sizes = np.exp2(np.arange(self.min_, self.max_, self.range_step_))
        grad = grad_huber_LR(beta, X, y, self.thres)
        iter_ = 0
        while True:
            if norm(grad) < self.stopping * self.thres:
                # print(f"grad_norm{norm(grad)}   Threshold: {self.stopping*self.thres}")
                break
            if iter_ >= self.max_iters_:
                # print(f"Iter: {iter_}")
                break
            iter_ = iter_ + 1
            huber_val = huber_LR(beta, X, y, self.thres)
            grad = grad_huber_LR(beta, X, y, self.thres)

            f_val = np.array(
                [huber_LR(beta - step * grad, X, y, self.thres) for step in step_sizes]
            )

            eta = step_sizes[np.argmin(f_val)]

            if logs_ is True:
                l2_err = norm(beta - true_beta)
                logs.append([huber_val, l2_err])
            beta = beta - eta * grad
        if logs_ is True:
            logs = np.array(logs)
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y, true_beta=true_beta)
        return norm(beta_hat - true_beta)


# Regressor 6: Least Trimmed Squares with Alternating Minimization (Baseline)
class LTS_Alt_Min(LinearRegressor):
    def __init__(self, steps=None, hard_thres_size=None):
        self.name = f"LTS_steps_{steps}_HT_{hard_thres_size}"
        self.steps = steps
        self.hard_thres_size = hard_thres_size

    def estimateBeta(self, X, y, true_beta=None, logs_=False, verbose=False):
        b = np.zeros_like(y)
        P = hat_matrix(X)
        if logs_ is True:
            logs = np.zeros((self.steps, 2))
        else:
            logs = None

        for i in range(self.steps):
            b = hard_thresholding(P.dot(b) + y - P.dot(y), self.hard_thres_size)
            if logs_:
                beta = leastsquares(X, y - b)
                sq_err = lts_error(X, y, beta, self.hard_thres_size)
                l2_err = norm(beta - true_beta)
                logs[i, 0] = sq_err
                logs[i, 1] = l2_err
        beta_hat = leastsquares(X, y - b)
        return beta_hat, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)



# Regressor 7: Least Trimmed Squares with Alternating Minimization and Early Stopping Criteria (Baseline)
# In contrast to the previous model (LTS_Alternating_Min), this model uses an early stopping criterion
# to stop the alternating minimization process when the norm of the difference between the current and previous beta coefficients
# falls below a certain threshold. This is done to stop early and save simulation time.
class LTS_Alt_Min_Early_Stopping(LinearRegressor):
    def __init__(self, hard_thres_size=None, stopping_thres=None):
        self.name = f"LTS_HT_{hard_thres_size}_stop_thres_{stopping_thres}"
        self.stopping_thres = stopping_thres
        self.hard_thres_size = hard_thres_size

    def estimateBeta(self, X, y, true_beta=None, logs_=False, verbose=False):
        b = np.zeros_like(y)
        P = hat_matrix(X)
        if logs_ is True:
            logs = []
        else:
            logs = None
        step = 0

        if logs_:
            beta = leastsquares(X, y - b)
            sq_err = lts_error(X, y, beta, self.hard_thres_size)
            l2_err = norm(beta - true_beta)
            logs.append([sq_err, l2_err])

        while True:
            b_prev = np.copy(b)
            b = hard_thresholding(P.dot(b) + y - P.dot(y), self.hard_thres_size)
            if logs_:
                beta = leastsquares(X, y - b)
                sq_err = lts_error(X, y, beta, self.hard_thres_size)
                l2_err = norm(beta - true_beta)
                logs.append([sq_err, l2_err])

            normie = norm(b - b_prev)

            if norm(b - b_prev) <= self.stopping_thres:
                break

        beta_hat = leastsquares(X, y - b)
        if logs_ is True:
            logs = np.array(logs)
        return beta_hat, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)



# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
# We now implement Huber regression and Least trimmed sqaures with covariate filtering.
# The idea is to filter out the covariates that are deemed outliers by the filtering algorithm.
# This is done to improve the robustness of the regression model to outliers in the data.


# --------------------------------------------------------------
# Regressor 8: Filtered Huber Regression with Gradient Descent
class Filter_HuberRegressionGD(LinearRegressor):
    # This is the version of the `HuberRegressionGD` that uses covariate filtering
    def __init__(
        self, thres=1, iters=None, step_size=None, beta_0=None, covFilter=None
    ):
        self.name = f"Hub_thres_{thres}_iters_{iters}_{covFilter.name}"
        self.thres = thres
        self.iters = iters
        self.step_size = step_size
        self.beta_0 = beta_0
        self.covFilter = covFilter
        self.huberRegressor = HuberRegressionGD(
            thres=thres, iters=iters, step_size=step_size, beta_0=beta_0
        )

    def estimateBeta(self, X, y, verbose=False, logs_=False, true_beta=None):
        filtered_w, _, _ = self.covFilter.estimateMean(X)
        filtered_X, filtered_y = X[filtered_w > 0], y[filtered_w > 0]
        beta, logs = self.huberRegressor.estimateBeta(
            filtered_X, filtered_y, verbose=verbose, logs_=logs_, true_beta=true_beta
        )
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y, true_beta=true_beta)
        return norm(beta_hat - true_beta)

# Regressor 9: Filtered Huber Regression with Gradient Descent and Early Stopping Criteria
class Filter_HuberRegression_GD_LineSearch_EarlyStopping(LinearRegressor):
    # This is the version of the `HuberRegression_GD_LineSearch_EarlyStopping` that uses covariate filtering
    def __init__(
        self,
        thres=1,
        stopping=None,
        beta_0=None,
        min_=-5,
        max_=5,
        range_step_=1,
        max_iters_=1e2,
        covFilter=None,
    ):
        self.name = f"Hub_LS_thres_{thres}_stopping_{stopping}_{covFilter.name}"
        self.thres = thres
        self.stopping = stopping
        self.beta_0 = beta_0
        self.min_ = min_
        self.max_ = max_
        self.range_step_ = range_step_
        self.func_diff = 1e-5
        self.covFilter = covFilter
        self.max_iters_ = max_iters_
        self.huberRegressor = HuberRegression_GD_LineSearch_EarlyStopping(
            thres=thres,
            stopping=stopping,
            beta_0=beta_0,
            min_=min_,
            max_=max_,
            range_step_=range_step_,
            max_iters_=max_iters_,
        )

    def estimateBeta(self, X, y, verbose=False, logs_=False, true_beta=None):
        filtered_w, _, _ = self.covFilter.estimateMean(X)
        filtered_X, filtered_y = X[filtered_w > 0], y[filtered_w > 0]
        beta, logs = self.huberRegressor.estimateBeta(
            filtered_X, filtered_y, verbose=verbose, logs_=logs_, true_beta=true_beta
        )
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y, true_beta=true_beta)
        return norm(beta_hat - true_beta)



# Regressor 10: Filtered Least Trimmed Squares with Alternating Minimization
class Filter_LTS_Alt_Min(LinearRegressor):
    # This is the version of the `LTS_Alt_Min` that uses covariate filtering
    def __init__(self, steps=None, hard_thres_size=None, covFilter=None):
        self.steps = steps
        self.hard_thres_size = hard_thres_size
        self.covFilter = covFilter
        self.LTS_regressor = LTS_Alt_Min(
            steps=steps, hard_thres_size=hard_thres_size
        )
        self.name = f"LTS_steps_{steps}_HT_{hard_thres_size}_{covFilter.name}"

    def estimateBeta(self, X, y, true_beta=None, logs_=False, verbose=False):
        filtered_w, _, _ = self.covFilter.estimateMean(X)
        filtered_X, filtered_y = X[filtered_w > 0], y[filtered_w > 0]
        beta, logs = self.LTS_regressor.estimateBeta(
            filtered_X, filtered_y, verbose=verbose, logs_=logs_, true_beta=true_beta
        )
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y)
        return norm(beta_hat - true_beta)



# Regressor 11: Filtered Least Trimmed Squares with Alternating Minimization and Early Stopping Criteria
class Filter_LTS_Alt_Min_Early_Stopping(LinearRegressor):
    # This is the version of the `LTS_Alt_Min_Early_Stopping` that uses covariate filtering       
    def __init__(self, hard_thres_size=None, stopping_thres=None, covFilter=None):
        self.name = (
            f"F_LTS_HT_{hard_thres_size}_stop_thres_{stopping_thres}_{covFilter.name}"
        )
        self.stopping_thres = stopping_thres
        self.hard_thres_size = hard_thres_size
        self.covFilter = covFilter
        self.LTS_regressor = LTS_Alt_Min_Early_Stopping(
            hard_thres_size=hard_thres_size, stopping_thres=stopping_thres
        )

    def estimateBeta(self, X, y, verbose=False, logs_=False, true_beta=None):
        filtered_w, _, _ = self.covFilter.estimateMean(X)
        filtered_X, filtered_y = X[filtered_w > 0], y[filtered_w > 0]
        beta, logs = self.LTS_regressor.estimateBeta(
            filtered_X, filtered_y, verbose=verbose, logs_=logs_, true_beta=true_beta
        )
        return beta, logs

    def estimateError(self, X, y, true_beta):
        beta_hat, _ = self.estimateBeta(X, y, true_beta=true_beta)
        return norm(beta_hat - true_beta)
