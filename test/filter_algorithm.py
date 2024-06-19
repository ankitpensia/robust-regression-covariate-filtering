import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import util


def filter_PBR(score,w):
    n = score.size
    prob = score/score.sum()
    sampled_point= np.random.choice(n,1,p=prob)
    w[sampled_point] = 0
    return w, sampled_point


def filter_covariates_deterministic(X,eps):
#     Implement the algorithm in the DKP paper.
    raise Exception

def filter_covariates_stochastic(X, steps=10):
#     The algorithm that removes a sample at every iteration
    n,d = X.shape
    w = np.ones(n)/n
    logs = {
            "removed_pts" : np.zeros(steps), 
            "est_error": np.zeros(steps), 
            "eigval":  np.zeros(steps),
            }
    for i in range(steps):
        eigval, eigvec, w_mean = util.leading_eigenvector(X,w,method="scipy_eigsh")
        proj =  (X - w_mean[None,:]).dot(eigvec)
        score = np.power(proj,2).flatten()
        top_heavy_elements_thresh = np.partition(score, steps)[-steps]
        # Compute the k-th largest number with k = steps.
        score[score < top_heavy_elements_thresh] = 0
        # Set the points with scores less than the threshold to be zero
        score[w == 0]= 0
        # w = 0 represents the points which have been removed
        # and thus their score should be 0
        w, removed_pt = filter_PBR(score,w)
        logs["eigval"][i] = eigval[0]
        logs["removed_pts"][i] = removed_pt[0]
        logs["est_error"][i] = np.linalg.norm(w_mean) 
    return w, logs



def plot_logs(logs,n,d,steps,outliers=None):
    arr = np.arange(steps)

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,15))
    ax1.plot(logs["eigval"])
    ax1.set_title('Largest eigenvalue across steps')
    ax1.set_xlabel('Num steps')
    ax1.set_yscale('log')

    gd_steps = logs["removed_pts"] >= n - outliers
    tot_out = np.cumsum(gd_steps)/n
    ax2.plot(arr, tot_out, label='Frac. of outliers removed')
    ax2.plot(arr, arr/n - tot_out , label='Frac. of inliers removed')
    ax2.set_title("Fraction of points removed")
    ax2.set_xlabel('Num steps')
    ax2.legend()

    ax3.plot( arr, logs["est_error"])
    ax3.set_yscale('log')
    ax3.set_ylabel("L-2 norm of error")

if __name__ == "__main__":
    n =1000
    outliers = 50
    d = 70
    X = np.random.randn(n,d)
    # scale=100
    # X[n-outliers:,:] = scale*np.random.randn(outliers,d)
    # scale=100
    X[n-outliers:,:] = 2*np.repeat(np.random.randn(d,1), outliers,axis=1).T
    steps = 4*outliers
    w, logs = filter_covariates_stochastic(X,steps)

    plot_logs(logs,n,d,steps,outliers)
    plt.show()