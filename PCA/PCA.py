from hashlib import new
from pickletools import uint8
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

data = scipy.io.loadmat("PCA/yalefaces.mat")
data = data["yalefaces"]
index = 8

n = 38
d = 2016


def k_by_threshold(thresh, l, d):
    k = d
    constrast = 1.00
    while constrast > thresh:
        k -= 1
        constrast = np.sum(l[:k]) / np.sum(l)
    return k + 1


empty_data = []
arr = np.zeros(d, dtype=np.uint8)
for _ in range(n):
    empty_data.append(arr)
x_data = np.array(empty_data)
for i in range(n):
    x_data[i] = np.reshape(data[:, :, i], (1, d))
x_data = np.transpose(x_data)
_x = np.mean(x_data, axis=1)
s = np.cov(x_data)
# s = np.reshape(np.zeros(d * d), (d, d))
# for i in range(n):
#     news = np.outer((x_data[i] - _x), (x_data[i] - _x))
#     s += news
#s /= n
lam, U = np.linalg.eig(s)
lam = np.real(lam)
U = np.real(U)
idx = lam.argsort()[::-1]
lam = lam[idx]
U = U[:, idx]
np.savetxt("sorted_eigenvalues.csv", lam)
plt.plot(np.linspace(1, d + 1, d), lam)

k_95 = k_by_threshold(0.95, lam, d)
print(f"the required principle componants are {k_95}, constrast {np.sum(lam[:k_95]) / np.sum(lam):.2f}")
print(f"dimensional reduction with k={k_95} is {(100* (d - k_95) / d):.2f} percent")

k_99 = k_by_threshold(0.99, lam, d)
print(f"the required principle componants are {k_99}, constrast {np.sum(lam[:k_99]) / np.sum(lam):.2f}")
print(f"dimensional reduction with k={k_99} is {(100* (d - k_99) / d):.2f} percent")

fig, axs = plt.subplots(4, 5)
comp_data = np.reshape(np.zeros(2016*20), (48, 42, 20))
for i in range(4):
    for j in range(5):
        k = k_99
        A = np.zeros((d, k + 1))
        A = np.c_[np.reshape(_x, (d, 1)), U[:, :k]]
        x_rel = np.ones(k+1)
        x_rel[1:] = x_data[0:k, i]
        newimage = np.matmul(A, x_rel)
        newimage = np.reshape(newimage, (48, 42))
        comp_data[:, :, i] = newimage
        axs[i, j].imshow(newimage, interpolation="nearest")
plt.show()
