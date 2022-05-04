# Import all relevant libraries and modules
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Read data from the csv file
data = pd.read_csv('Task2 - dataset - dog_breeds.csv')

# shuffle data rows
data = data.sample(frac=1)

""" # show the scatter plot of the data
x_kmeans = data.iloc[:, 0]
print(x_kmeans)
y_kmeans = data.iloc[:, 1]

plt.clf()
plt.scatter(x_kmeans,y_kmeans)
plt.savefig('kmeans_scatter.png')
plt.show()

# plot figure
plt.show() """

# Calculate euclidean distance between vectors 1 and 2
def compute_euclidean_distance(vec_1, vec_2):
    distance = 0

    # For each element of vector 1 and 2
    for a, b in zip(vec_1, vec_2):
        # Add difference squared to total distance
        distance = distance + ((a - b) ** 2)
    
    # Square root distance
    distance = math.sqrt(distance)

    return distance

# Randomly initialise the centroids based on k
def initialise_centroids(dataset, k):
    centroids = 0
    # Minimum value of the dataset, rounded down to int
    min = np.floor(np.array(np.amin(dataset)))
    # Maximum value of the dataset, rounded up to int
    max = np.ceil(np.array(np.amax(dataset)))

    # Initialise first centroid as random 4D vector
    centroids = np.array([random.randint(min[0], max[0]), random.randint(min[1], max[1]), random.randint(min[2], max[2]), random.randint(min[3], max[3])])

    # For each number of kmeans
    for i in range(k - 1):
        # Create new centroid
        centroid = np.array([random.randint(min[0], max[0]), random.randint(min[1], max[1]), random.randint(min[2], max[2]), random.randint(min[3], max[3])])
        # Append to previously created matrix
        centroids = np.vstack([centroids, centroid])

    return centroids

# Recalculates the centroid position based on plot
def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids

def assignPoints(dataset, centroids):
    # Labels 0,1,2
    clusters = np.zeros(len(dataset))
    # Assign each value to its closest cluster
    for i in range(len(dataset)):
        distances = compute_euclidean_distance(dataset[i], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters   

# Cluster the data into k groups
def kmeans(dataset, k):
    sumDistances = []
    # Create centroids based on k
    centroids = initialise_centroids(dataset, k)
    # Initialise array
    assignmentsOld = np.empty(dataset.shape[0], dtype=int)
    Exit = False
    while Exit == False:
        # Assign all of the points to a centroid
        assignmentsNew = assignPoints(dataset, centroids)
        # If no points are assigned to a different centroid, break loop
        if np.array_equal(assignmentsOld, assignmentsNew):
            Exit = True
        # If not, store the new assignments
        else:
            assignmentsOld = assignmentsNew
        # For each centroid
        for item in centroids:
            #if the centroid is empty, rerun the process
            if np.count_nonzero(item) == 0:
                return kmeans(dataset, k)
        # Re-calculate the centroids as the cluster mean
        centroids = recalculate_centroids(dataset, assignmentsNew, k)
        # Append the array with the sum of the distance
        sumDistances.append(np.array(dataset, centroids, assignmentsOld)) 
    # Return the current assignments as the final clusters
    cluster_assigned = assignmentsNew
    return centroids, cluster_assigned


# Call k-means on the loaded data
K = 2
centroids, cluster_assigned = kmeans(data, K)
colors = ['k', 'b', 'g', 'r']

plt.clf()
for key in cluster_assigned.keys():
    current_cluster = np.array(cluster_assigned[key])
    
    plt.scatter(current_cluster[:,0], current_cluster[:,1], c = colors[key])
    
    current_centroid = centroids[key]
    
    plt.plot(current_centroid[0], current_centroid[1], 'gX', c='r', ms = 15)
#plt.show()