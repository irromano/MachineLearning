import numpy as np
from matplotlib import pyplot as plt


n = 20000
k = 10000
dc = 6
de = 3

passes = 2
blocks = 500


def init_U_messages(s, p):
    r0 = 1 + (1 - 2*p) ** (dc - 1)
    r1 = 1 - (1 - 2*p) ** (dc - 1)

    if s:
        temp = r0
        r0 = r1
        r1 = temp
    return np.full((dc), np.log(r0/r1))


def update_V_messages(V, U, en_cn, cn_en, p):
    for errorNode in range(en_cn.shape[0]):
        for i in range(en_cn.shape[1]):
            V[errorNode, i] = np.log((1-p) / p)
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
    return (1 - (1 - 2*pv) ** (dc - 1)) / 2


def compute_pv(pu, p, de):
    return (1 - p) * (pu) ** (de - 1) + p * (1 - (1 - pu) ** (de - 1))


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

prob_iterations = 11
probs = np.linspace(0.01, 0.21, prob_iterations)
errorRates = np.zeros(prob_iterations, dtype=np.longdouble)
predictedErrorRates = errorRates.copy()

for probability in range(prob_iterations):
    p = probs[probability]
    totalErrors = 0
    for b in range(blocks):
        r = np.zeros(n, dtype=int)
        for i in range(len(r)):
            r[i] = np.random.choice([0, 1], 1, p=[1-p, p])
        s = np.matmul(r, H.transpose()) % 2

        # Initialize V messages
        V = np.full((n, de), np.log((1 - p) / p))
        # U messages are set to 1 by default
        U = np.ones((n-k, dc))

        for checkNode in range(U.shape[0]):
            U[checkNode] = init_U_messages(s[checkNode], p)

        for i in range(passes):
            V = update_V_messages(V, U, errorNode_to_checkNode,
                                  checkNode_to_errorNode, p)
            U = update_U_messages(U, V, s, checkNode_to_errorNode,
                                  errorNode_to_checkNode)

        d = np.full(n, np.log((1-p) / p))
        #d = np.zeros(n)
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
    errorRates[probability] = errorRate
    print(f"BER at p={probs[probability]} is {errorRate}")

    # Calculating predicted Error Rate
    pv = p
    for i in range(passes):
        pu = compute_pu(pv, dc)
        pv = compute_pv(pu, p, de)
    predictedErrorRates[probability] = pv / de
    print(f"PER at p={probs[probability]} is {pv / de}")


plt.xlabel("Probability of bit flip")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(probs, errorRates, color="red", label="BER")
plt.plot(probs, predictedErrorRates, color="blue", label="Prediction of BER")
plt.legend()
plt.show()
