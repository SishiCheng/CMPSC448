import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = np.loadtxt('iris.data', delimiter = ",", dtype='str')
#print(data)

temp = pd.DataFrame(data, columns=list('abcde'))
temp['e'] = [0 if item == 'Iris-setosa' else 1 if item == 'Iris-versicolor' else 2 for item in temp['e']]
temp = temp.astype(float)
x1 = temp['a']/temp['b']
x2 = temp['c']/temp['d']

new_data = pd.DataFrame()
new_data['x1'] = x1
new_data['x2'] = x2
new_data['class'] = temp['e']

new_data.plot.scatter('x1', 'x2', c='class', colormap='viridis')
plt.show()


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    # initialize k centroids
    centroids = np.zeros((k, X.shape[1]))
    #print(centroids[0])
    centroids[0] = X[np.random.randint(X.shape[0])]
    #print(X[0])
    #centroids[0].append(X[0])
    for i in range(1, k):
        # compute distances from centroids
        distances = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            distances[j] = np.linalg.norm(X[j] - centroids[i-1])
        # compute probabilities
        probabilities = distances / np.sum(distances)
        # choose centroid
        centroids[i] = X[np.random.choice(X.shape[0], 1, p=probabilities)]
        print(X.shape[0])
    return centroids


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    data_map =[]
    for i in range(len(X)):
        cluster = None
        min_dist = np.inf
        for c in C:
            # Calculate distance between data point and cluster center
            dist = np.linalg.norm(X[i]-c)
            if dist < min_dist:
                min_dist = dist
                cluster = c
        data_map.append(cluster)
    return data_map


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    data_map = assign_data2clusters(X,C)
    total_dist = 0
    for c in centers:
        for item in data_map:
            dist = 0
            if item[1][0] == c[0] and item[1][1] == c[1]:
                for j in range(len(c)):
                    dist += (c[j] - item[0][j])**2
                total_dist += dist
    return total_dist


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    objective_values: array, shape (max_iter)
        The objective value at each iteration
    """
    # initialize centroids
    centroids = k_init(X, k)
    # initialize cluster assignment
    cluster_assignment = np.zeros(X.shape[0])
    # initialize cluster centers
    cluster_centers = np.zeros((k, X.shape[1]))
    # initialize cluster counts
    cluster_counts = np.zeros(k)
    # initialize cluster centers
    cluster_centers = centroids
    # initialize cluster counts
    cluster_counts = np.zeros(k)
    # initialize iteration counter
    iter_count = 0
    # initialize convergence flag
    converged = False
    # run kmeans++ algorithm
    while not converged:
        # update iteration counter
        iter_count += 1
        # update cluster assignment
        for i in range(X.shape[0]):
            # compute distances from centroids
            distances = np.zeros(k)
            for j in range(k):
                distances[j] = np.linalg.norm(x[i] - centroids[j])
            # compute probabilities
            probabilities = distances / np.sum(distances)
            # choose centroid
            cluster_assignment[i] = np.random.choice(k, 1, p=probabilities)
        # update cluster centers
        for i in range(k):
            # compute cluster counts
            cluster_counts[i] = np.sum(cluster_assignment == i)
            # compute cluster centers
            cluster_centers[i] = np.sum(x[cluster_assignment == i], axis=0) / cluster_counts[i]
        # check for convergence
        if np.array_equal(centroids, cluster_centers):
            converged = True
        else:
            centroids = cluster_centers
    return centroids


cluster_data = pd.DataFrame()
cluster_data['x1'] = x1
cluster_data['x2'] = x2

#run with k=1,2,3,4,5
acc = []
centers = k_means_pp(cluster_data, 1, 50)
acc.append(compute_objective(cluster_data, centers))

centers = k_means_pp(cluster_data, 2, 50)
acc.append(compute_objective(cluster_data, centers))

centers = k_means_pp(cluster_data, 3, 50)
acc.append(compute_objective(cluster_data, centers))

centers = k_means_pp(cluster_data, 4, 50)
acc.append(compute_objective(cluster_data, centers))

centers = k_means_pp(cluster_data, 5, 50)
acc.append(compute_objective(cluster_data, centers))

plt.plot([1,2,3,4,5], acc)
plt.show()
plt.clf()

#my best was k=5 so now ill change the number fo iterations with k=5
acc = []
centers = k_means_pp(new_set, 5, 1)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 20)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 50)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 100)
acc.append(compute_objective(new_set, centers))

centers = k_means_pp(new_set, 5, 200)
acc.append(compute_objective(new_set, centers))

plt.plot([1,20,50,100,200], acc)
plt.show()
plt.clf()

#plot with data colored by cluster
centers = k_means_pp(new_set, 5, 200)
data_map = assign_data2clusters(new_set, centers)
label = []
for i in range(len(data_map)):
    if data_map[i][1] == centers[0]:
        label.append(0)
    elif data_map[i][1] == centers[1]:
        label.append(1)
    elif data_map[i][1] == centers[2]:
        label.append(2)
    elif data_map[i][1] == centers[3]:
        label.append(3)
    elif data_map[i][1] == centers[4]:
        label.append(4)
plt.scatter(x1, x2, c=label)
plt.show()
