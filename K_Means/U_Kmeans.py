

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Long list of colors for plotting large amount of datapoints
allColors = np.concatenate((np.array(list(mcolors._colors_full_map.values())), np.array(list(mcolors._colors_full_map.values())), np.array(list(mcolors._colors_full_map.values()))))

#############################################################################
# Generate three seperate clusters
#############################################################################
n = 1000
x1 = np.random.randn(n, 2) + np.matlib.repmat([4, 4], n, 1)
x2 = np.random.randn(n, 2) + np.matlib.repmat([4, 12], n, 1)
x3 = np.random.randn(n, 2) + np.matlib.repmat([10, 8], n, 1)
data = np.concatenate((x1, x2, x3), axis=0)
n = data.shape[0]

# U Kmeans variables
epilson = 0.1
c = n
a = data.copy()


def get_centroids(data, K, a):
    classes = np.zeros(data.shape[0], dtype=int)
    euc = np.zeros((K, data.shape[0]))
    for i in range(K):

        euc[i] = np.sum(np.square(data - a[i]), axis=1)
    classes = np.argmin(euc, axis=0)

    for i in range(K):
        a[i] = np.mean(data[classes == i], axis=0)
        # if len(data[classes == i]) > 0:
        #     Centroids[i] = np.mean(data[classes == i], axis=0)
    return a, classes

#############################################################################
#                       Main Function
#############################################################################
# Hyper-parameters
K_iter = 3000
epochs = 2

# Initialize the centroids
data_min = np.min(data)
data_max = np.max(data)
print('the min is %.2f' % (data_min))
print('the max is %.2f' % (data_max))
for K in range(K_iter, K_iter + 1):
    #a = np.random.randint(low=data_min, high=data_max, size=(K, 2))
    #print(a)

    # Define the centroids vector for visualizing the training procedure
    centroids_vector = np.zeros((epochs, K, 2))

    # Training loop for K-means clustering
    for epoch in range(epochs):
        a, classes = get_centroids(data, K, a)
        # classes = np.zeros(n * 3, dtype=int)
        centroids_vector[epoch] = a
        
    #############################################################################
    # Visualizing the dataset and the learned K-mean model
    #group_colors = ['skyblue', 'coral', 'lightgreen']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    group_colors = prop_cycle.by_key()['color']
    colors = [allColors[j] for j in classes]
    print(len(allColors))

    fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    ax = _axs.flatten()

    ax[0].scatter(data[:, 0], data[:, 1])
    ax[0].set_title('The orignial dataset with 3 clusters')


    ax[1].scatter(data[:, 0], data[:, 1], color=colors, alpha=0.5)
    # ax[1].scatter(centroids_vector[epochs-1][:, 0], centroids_vector[epochs-1]
    #             [:, 1], color=allColors[:K], marker='o', lw=2)

    # l_tick = [ax[1].scatter(centroids_vector[:, i, 0], centroids_vector[:, i, 1],
    #                     c=range(epochs), vmin=1, vmax=epochs, cmap='autumn', marker='v') for i in range(K)]

    # for i in range(K):
    #     ax[1].plot(centroids_vector[:, i, 0], centroids_vector[:, i, 1], alpha=0.5, color='k')
    ax[1].set_xlabel('$x_0$')
    ax[1].set_ylabel('$x_1$')
    # cbar = fig.colorbar(l_tick[K-1], ax=ax[1])
    # cbar.set_label('Epoch')
    ax[1].set_title(f'The final dataset with {K} clusters')


    plt.show()
