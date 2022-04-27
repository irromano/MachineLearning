import numpy as np
import scipy.stats as stats
from scipy import signal
from matplotlib import pyplot as plt


n = 20000
k = 10000
dc = 6
de = 3

passes = 4
blocks = 500


def init_U_messages(U, p, s, cn_en):
    for checkNode in range(cn_en.shape[0]):
        for i in range(cn_en.shape[1]):
            r0 = 1
            for j in range(cn_en.shape[1]):
                if j == i:
                    continue
                errorNode = cn_en[checkNode, j]
                r0 *= 1 - 2*p[errorNode]
            r1 = r0
            r0 = 1 + r0
            r1 = 1 - r1
            if s[checkNode]:
                temp = r0

                r0 = r1
                r1 = temp
            U[checkNode, i] = np.log(r0/r1)
    return U


def update_V_messages(V, U, en_cn, cn_en, p):
    for errorNode in range(en_cn.shape[0]):
        for i in range(en_cn.shape[1]):
            V[errorNode, i] = p[errorNode]
            for j in range(en_cn.shape[1]):
                if j == i:
                    continue
                checkNode = en_cn[errorNode, j]
                for k in range(cn_en.shape[1]):
                    if cn_en[checkNode, k] == errorNode:
                        V[errorNode, i] += U[checkNode, k]

    return V


def update_U_messages(U, V, s, cn_en, en_cn):
    for checkNode in range(cn_en.shape[0]):
        for i in range(cn_en.shape[1]):
            U[checkNode, i] = 1
            for j in range(cn_en.shape[1]):
                if j == i:
                    continue
                errorNode = cn_en[checkNode, j]
                for k in range(en_cn.shape[1]):
                    if en_cn[errorNode, k] == checkNode:
                        U[checkNode, i] *= np.tanh(V[errorNode, k] / 2)
            if U[checkNode, i] == 1.0:
                U[checkNode, i] = 0.999999
            if U[checkNode, i] == -1.0:
                U[checkNode, i] = -0.999999
            U[checkNode, i] = np.arctanh(
                U[checkNode, i]) * 2 * (-1) ** s[checkNode]
    return U


def compute_pu(pv, dc):
    pu = signal.fftconvolve(np.tanh(pv/2), np.tanh(pv/2), "same")
    pu = np.arctanh(pu) * 2
    for i in range(dc - 2):
        pu = signal.fftconvolve(np.tanh(pu/2), np.tanh(pv/2), "same")
        pu = np.arctanh(pu) * 2
    return pu


def compute_pv(pu, li, de):
    pv = li.copy()
    for i in range(de - 1):
        pv = signal.fftconvolve(pv, pu, "same")
    return pv


# Load Tanner Graph from files
# checkNode_to_errorNode = np.loadtxt("LDPC_Data/checkNode_to_errorNode.csv", dtype=int)
# errorNode_to_checkNode = np.loadtxt("LDPC_Data/errorNode_to_checkNode.csv", dtype=int)


# Creates new Parity Matrix. Has a chance of crashing
while True:
    try:
        checkNode_to_errorNode = []
        errorNode_to_checkNode = [[] for _ in range(n)]
        errorNodeList = np.arange(n)
        for checkNode in range(n-k):
            arr = np.random.choice(errorNodeList, dc, replace=False)
            checkNode_to_errorNode.append(arr)
            for errorNode in arr:
                errorNode_to_checkNode[errorNode].append(checkNode)
                if len(errorNode_to_checkNode[errorNode]) >= de:
                    errorNodeList = np.delete(
                        errorNodeList, np.where(errorNodeList == errorNode))
        # np.savetxt("checkNode_to_errorNode.csv", checkNode_to_errorNode)
        # np.savetxt("errorNode_to_checkNode.csv", errorNode_to_checkNode)

        checkNode_to_errorNode = np.array(checkNode_to_errorNode)
        errorNode_to_checkNode = np.array(errorNode_to_checkNode)
        break
    except ValueError:
        print("Creating new Tanner Graph failed. Trying again ...")

# Build Parity Matrix from tanner graph arrays
H = np.zeros((n-k, n), dtype=int)
for checkNode in range(len(checkNode_to_errorNode)):
    for errorNode in checkNode_to_errorNode[checkNode]:
        H[checkNode, errorNode] = 1

Eb_iterations = 13
Eb_list = np.linspace(0.1, 1.3, Eb_iterations)
errorRates = np.zeros(Eb_iterations, dtype=np.longdouble)
predictedErrorRates = errorRates.copy()

for energy in range(Eb_iterations):
    eb = Eb_list[energy]
    totalErrors = 0
    for b in range(blocks):

        w = np.random.normal(eb, 1.0, size=n)  # Soft recieved observation
        llr = 2 * w  # np.exp(-(w - eb)**2) / np.exp(-(w + eb)**2)  # Log likelyhood
        p = np.exp(-(w + eb)**2) / (np.exp(-(w + eb)**2) + np.exp(-(w - eb)**2))
        r = llr.copy()
        r[r > 0] = 0  # ones represent -es sqrt
        r[r < 0] = 1  # zeros represent es sqrt
        errors = np.sum(r)
        s = np.matmul(r, H.transpose()) % 2

        # Initialize V messages
        V = np.full((de, n), llr)
        V = V.transpose()
        # U messages are set to 1 by default
        U = np.ones((n-k, dc))
        U = init_U_messages(U, p, s, checkNode_to_errorNode)

        for i in range(passes):
            V = update_V_messages(V, U, errorNode_to_checkNode,
                                  checkNode_to_errorNode, p)
            U = update_U_messages(U, V, s, checkNode_to_errorNode,
                                  errorNode_to_checkNode)

        #d = np.full(n, np.log((1-p) / p))
        d = llr.copy()
        for errorNode in range(d.shape[0]):
            for i in range(errorNode_to_checkNode.shape[1]):
                checkNode = errorNode_to_checkNode[errorNode, i]
                for j in range(checkNode_to_errorNode.shape[1]):
                    if checkNode_to_errorNode[checkNode, j] == errorNode:
                        d[errorNode] += U[checkNode, j]

        d = (d < 0) * 1.0
        errors = d != r
        totalErrors += np.sum(errors)
    errorRate = totalErrors / (n * blocks)
    errorRates[energy] = errorRate
    print(f"BER at Eb={Eb_list[energy]} is {errorRate}")

    # Calculating predicted Error Rate
    normal_dist = stats.norm(loc=eb, scale=1)
    delta = 1e-4
    big_grid = np.arange(-10, 10, delta)
    li = normal_dist.pdf(big_grid)*delta
    pv = li.copy()
    for i in range(2):

        pu = compute_pu(pv, dc)
        pv = compute_pv(pu, li, de)
    D_pred = li.copy()
    for i in range(de):
        D_pred = signal.fftconvolve(D_pred, pu, "same")

    predError = np.trapz(D_pred[0:(big_grid.size // 2)]) / np.trapz(D_pred)
    predictedErrorRates[energy] = predError
    print(f"PER at Eb={Eb_list[energy]} is {predError}")


plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Eb_list, errorRates, color="red", label="BER")
plt.plot(Eb_list, predictedErrorRates, color="blue", label="Prediction of BER")
plt.legend()
plt.show()
