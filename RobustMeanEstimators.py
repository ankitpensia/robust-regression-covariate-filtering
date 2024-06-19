# This file implements the filtering algorithm for robust mean estimation.
import numpy as np
from abc import ABC, abstractmethod
import util


class MeanEstimator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def estimateMean(self, X):
        pass

    @abstractmethod
    def estimateError(self, X, true_mean=None):
        pass


class SampleMean(MeanEstimator):
    def __init__(self):
        self.name = "SampleMean"

    def estimateMean(self, X, true_mean=None):
        return X.mean(axis=0)

    def estimateError(self, X, true_mean=None):
        mu_hat = self.estimateMean(X)
        if true_mean is not None:
            return np.linalg.norm(mu_hat - true_mean)
        else:
            return np.linalg.norm(mu_hat)


class StochasticFilterPBR(MeanEstimator):
    def __init__(self, steps=None):
        self.steps = steps
        self.name = f"F_steps_{steps}"

    def set_steps(self, steps):
        self.steps = steps

    def estimateMean(self, X, true_mean=None):
        n, d = X.shape
        w = np.ones(n) / n
        if self.steps == 0:
            return w, util.weighted_mean(X, w), None
        if true_mean is None:
            true_mean = np.zeros(d)
        logs = {
            "removed_pts": np.zeros(self.steps),
            "est_error": np.zeros(self.steps),
            "eigval": np.zeros(self.steps),
        }
        for i in range(self.steps):
            eigval, eigvec, w_mean = util.leading_eigenvector(
                X, w, method="scipy_eigsh"
            )
            proj = (X - w_mean[None, :]).dot(eigvec)
            score = np.power(proj, 2).flatten()
            score[w == 0] = 0
            # w = 0 represents the points which have been removed
            # and thus their score should be 0
            w, removed_pt = filter_PBR(score, w)
            # Modified weighted mean
            w_mean = util.weighted_mean(X, w)
            logs["eigval"][i] = eigval
            logs["removed_pts"][i] = removed_pt
            logs["est_error"][i] = np.linalg.norm(w_mean - true_mean)
        return w, w_mean, logs

    def estimateError(self, X, true_mean=None):
        w, w_mean, logs = self.estimateMean(X, true_mean=true_mean)
        return logs["est_error"][-1]


def filter_PBR(score, w):
    n = score.size
    prob = score / score.sum()
    sampled_point = np.random.choice(n, 1, p=prob)
    w[sampled_point] = 0
    return w, sampled_point
