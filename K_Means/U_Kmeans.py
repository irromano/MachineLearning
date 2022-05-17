

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Long list of colors for plotting large amount of datapoints
allColors = np.concatenate((np.array(list(mcolors._colors_full_map.values())), np.array(list(mcolors._colors_full_map.values())), np.array(list(mcolors._colors_full_map.values()))))

#############################################################################
# Generate three seperate clusters
#############################################################################
n = 65
sig = np.identity(2) * 0.4
x1 = np.random.multivariate_normal([5, 2], sig, n)
x2 = np.random.multivariate_normal([3, 4], sig, n+5)
x3 = np.random.multivariate_normal([8, 4], sig, n)
x4 = np.random.multivariate_normal([6, 6], sig, n)
x5 = np.random.multivariate_normal([10, 8], sig, n+5)
x6 = np.random.multivariate_normal([7, 10], sig, n)
data = np.concatenate((x1, x2, x3, x4, x5, x6), axis=0)
n = data.shape[0]
d = data.shape[1]

# U Kmeans variables
epilson = 0.1
k = n
a = data.copy() #np.around(data.copy(), 2)
alpha = np.full(k, 1 / k)   #Proportion of each cluster
gamma = 1
beta = 1


def update_classes(data, K, a, gamma, alpha):
    classes = np.zeros((n, K), dtype=int)
    euc = np.zeros((K, n))
    alpha_log = np.log(alpha)
    for i in range(K):
        euc[i] = np.around(np.sum(np.square(data - a[i]), axis=1) - gamma * alpha_log[i], 3)
    z = np.argmin(euc, axis=0)
    for i in range(n):
        classes[i, z[i]] = 1
    return classes

def update_gamma(K):
    return np.exp(-K/250)

def update_alpha(alpha, classes, beta, gamma):
    oldAlpha = alpha.copy()
    oldAlpha_log = np.log(oldAlpha)
    entropy = np.sum(oldAlpha * oldAlpha_log)
    z = np.mean(classes, axis=0)
    for i in range(alpha.shape[0]):
        alpha[i] = z[i] + (beta / gamma) * oldAlpha[i] * (oldAlpha_log[i] - entropy)
    # if np.ptp(oldAlpha_log):
    #     for i in range(alpha.shape[0]):
    #         alpha[i] = z[i] + (beta / gamma) * oldAlpha[i] * (oldAlpha_log[i] - entropy)
    # else:
    #     alpha = z
    #print(f"{alpha[0]} vs {oldAlpha[0]}")
    return alpha, oldAlpha

def update_beta(alpha, oldAlpha, K, classes, iter):
    entropy = 1 - np.max(np.sum(classes, axis=0)) / (-1 * np.max(oldAlpha * np.sum(np.log(oldAlpha))))
    longN = 1 if 1 < (1 / iter ** ((d - 2) // 2)) else (1 / iter ** ((d - 2) // 2))
    alphaDiff = np.mean(np.exp(-longN * n * np.absolute(oldAlpha - alpha)))

    return alphaDiff if alphaDiff < entropy else entropy

def remove_clusters(alpha, k, classes):
    keep_ind = alpha >= 1 / n
    if np.max(keep_ind) < True:
        keep_ind = np.full(keep_ind.shape, True)
    newAlpha = alpha[keep_ind]
    #a = a[keep_ind]
    k = newAlpha.shape[0]

    newAlpha = newAlpha / np.sum(newAlpha)
    classes = classes[:, keep_ind]

    
    return newAlpha, k, classes, keep_ind


def update_clusters(data, k, classes, olda):
    a = np.zeros((k, d))
    z = np.sum(classes, axis=0)

    for c in range(k):
        if keep_ind[c]:
            for i in range(n):
                a[c] += data[i] * classes[i, c]
            if z[c] > 0:
                a[c] /= z[c]
            else:
                a[c] = olda[c]

    a[np.sum(a, axis=1) == 0] = olda[np.sum(a, axis=1) == 0]
    # for i in range(n):
    #     a[classes_ind[i]] += data[i]  / z[i]
    return a

#############################################################################
#                       Main Function
#############################################################################

# Initialize the centroids
data_min = np.min(data)
data_max = np.max(data)
print('the min is %.2f' % (data_min))
print('the max is %.2f' % (data_max))
K_NotFound = True
iter = 0
centroids_vector = np.zeros((iter, k, 2))
k_array = np.full(1, k)
cost_array = np.full(1, -1)
classes= update_classes(data, k, a, gamma, alpha)
while (K_NotFound):
    gamma = update_gamma(k)
    alpha, oldAlpha = update_alpha(alpha, classes, beta, gamma)
    beta = update_beta(alpha, oldAlpha, k, classes, iter)
    alpha, k, classes, keep_ind = remove_clusters(alpha, k, classes)
    classes = update_classes(data, k, a[keep_ind], gamma, alpha)
    k_array = np.append(k_array, k)
    if (iter >= 60 and k_array[iter - 60] - k_array[iter] == 0):
        beta = 0
    new_a = update_clusters(data, k, classes, a[keep_ind])
    cost = np.max(np.concatenate((np.sqrt(np.sum(np.square(new_a - a[keep_ind]), axis=1)), np.sqrt(np.sum(np.square(a[np.logical_not(keep_ind)]), axis=1)))))
    print(f"Test difference is {cost}")
    if (cost < epilson and iter > 1):
        K_NotFound = False
    a = new_a
    iter += 1
    print(f"Current K={k}")
        
#############################################################################
# Visualizing the dataset and the learned K-mean model
#group_colors = ['skyblue', 'coral', 'lightgreen']
    z = np.argmax(classes, 1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    group_colors = prop_cycle.by_key()['color']
    colors = [allColors[j] for j in z]
    if k < 100:
        print(a)

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
    ax[1].set_title(f'The final dataset with {k} clusters')


plt.show()
