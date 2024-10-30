import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats
from matplotlib.patches import Ellipse

group_means = np.array([[-5.0, -0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, +0.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
    # Euclodian distance is way to go
    distance_to_means = dt.cdist(X, means, 'euclidean')
    # choose min distance
    means_clusters = np.argmin(distance_to_means, axis=1)
    covariances=np.zeros((K,X.shape[1],X.shape[1]))
    priors=np.zeros(K)
    for i in range(K):
        # lets get data points
        data_points = X[means_clusters == i]
        # priors
        priors[i] = data_points.shape[0] / X.shape[0]
        # Covariances 
        covariances[i] = np.dot((data_points - means[i]).T, (data_points - means[i])) / data_points.shape[0]
    # your implementation ends above
    return(means, covariances, priors)


means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    n, d = X.shape  
    assignments = np.zeros((n, K))
    def normal_distribution(x, mu, sigma):
        d = x.shape[1]  # Dimensionan of data points
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)
        log_prob = []

        for i in range(x.shape[0]):  # Loop over each data 
            diff = (x[i, :] - mu).reshape(-1, 1)
            exponent = -0.5 * np.dot(np.dot(diff.T, inv_sigma), diff)
            log_prob.append(-0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(det_sigma) + exponent.item())

        return np.array(log_prob).reshape(x.shape[0], 1)
        
    
    #E-step
    for iteration in range (100):
        for k in range(K):
            assignments[:, k] = priors[k] * np.exp(normal_distribution(X, means[k], covariances[k]).flatten())

        assignments_sum = assignments.sum(axis=1).reshape(-1, 1)
        assignments /= assignments_sum
        
        #m step
        Nk = assignments.sum(axis=0)

        for k in range(K):
            priors[k] = Nk[k] / n
            means[k] = (assignments[:, k].reshape(-1, 1) * X).sum(axis=0) / Nk[k]

            cov_k = np.zeros((d, d))
            for i in range(n):
                diff = (X[i, :] - means[k]).reshape(-1, 1)
                cov_k += assignments[i, k] * np.dot(diff, diff.T)
            cov_k /= Nk[k]

            covariances[k] = cov_k
        '''
        # Debugging
        print(f"Iteration {iteration + 1}:")
        print("Means:", means)
        print("Priors:", priors)
        '''
    
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    colors = ["red", "blue", "green", "purple"]
    
    x = np.linspace(-8, 8, 1600)  # 0.01 increment 
    y = np.linspace(-8, 8, 1600)  # 0.01 increment 
    X_grid, Y_grid = np.meshgrid(x, y)
    plot_points = np.dstack((X_grid, Y_grid))

    for k in range(K):
        data_points = X[np.argmax(assignments, axis=1) == k]
        plt.scatter(data_points[:, 0], data_points[:, 1], color=colors[k], s=5)
        
        given_gaussian_line = stats.multivariate_normal.pdf(plot_points, mean=group_means[k], cov=group_covariances[k])
        plt.contour(X_grid, Y_grid, given_gaussian_line, levels=[0.01], colors="black", linestyles="dashed")

        found_gaussian_line = stats.multivariate_normal.pdf(plot_points, mean=means[k], cov=covariances[k])
        plt.contour(X_grid, Y_grid, found_gaussian_line, levels=[0.01], colors=colors[k], linestyles="solid")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    #this is a very regular plot code, idea is taken from stack overflow.
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)