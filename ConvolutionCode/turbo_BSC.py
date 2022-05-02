import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import default_rng

import Conv_Trellis as Conv

# Constants
BLOCK_LENGTH = 16000
CONVOS = 2
MEMORY = 3
STATE_COUNT = (2 ** MEMORY)
ITERATIONS = 10
BLOCKS = 20
CYCLES = 1

Probability = np.linspace(0.031, 0.121, num=ITERATIONS)
BER_data0 = np.zeros(ITERATIONS, dtype=np.double)
BER_data1 = np.zeros(ITERATIONS, dtype=np.double)
BER_data2 = np.zeros(ITERATIONS, dtype=np.double)

# initialize randomizer
rng = default_rng()

for iter in range(ITERATIONS):
    # BER_0 = np.zeros(BLOCKS, dtype=np.double)
    BER_1 = np.zeros(BLOCKS, dtype=np.double)
    for block in range(BLOCKS):
        # Declaring uncoded and coded matrixes
        uncoded1 = rng.choice(2, BLOCK_LENGTH + MEMORY)  # np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)
        uncoded1[BLOCK_LENGTH:] = 0

        # Generating random permutation
        pi = rng.permutation(BLOCK_LENGTH + MEMORY)

        uncoded2 = uncoded1[pi].copy()

        # Inverse permutation
        # uncoded3 = np.zeros(BLOCK_LENGTH + MEMORY)
        # uncoded3[pi] = uncoded2

        coded1 = np.zeros((BLOCK_LENGTH + MEMORY, 2), dtype=np.double)
        coded2 = coded1.copy()

        # encoding uncoded
        d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
        D = np.zeros((BLOCK_LENGTH + MEMORY, MEMORY), dtype=np.int16)
        for i in range(BLOCK_LENGTH):
            D[i] = d.copy()
            coded1[i] = Conv.conv(uncoded1[i], d, Conv.EncodingType.BSC)

        # encoding permutation
        d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
        D = np.zeros((BLOCK_LENGTH + MEMORY, MEMORY), dtype=np.int16)
        for i in range(BLOCK_LENGTH):
            D[i] = d.copy()
            coded2[i] = Conv.conv(uncoded2[i], d, Conv.EncodingType.BSC)

        # Using the last 3 bits to reset state to [ 0, 0, 0 ]
        for i in range(BLOCK_LENGTH, BLOCK_LENGTH + MEMORY):
            D[i] = d.copy()
            resetbit = (d[1] + d[2]) % 2
            uncoded1[i] = resetbit
            coded1[i] = Conv.conv(resetbit, d, Conv.EncodingType.BSC)

        # Copying u bit of coded1 to coded2
        coded2[:, 0] = coded1[:, 0]

        coded_observation1 = coded1.copy()
        coded_observation2 = coded2.copy()

        # introducing noise
        p = Probability[iter]
        coded_observation1 = coded1.copy()
        error1 = rng.choice(2, (BLOCK_LENGTH + MEMORY, 2), p=[1-p, p])
        coded_observation1 = (coded_observation1 + error1) % 2
        coded_observation2 = coded2.copy()
        error2 = rng.choice(2, (BLOCK_LENGTH + MEMORY, 2), p=[1-p, p])
        coded_observation2 = (coded_observation2 + error2) % 2

        trel_1 = Conv.Trellis(Conv.EncodingType.BSC)
        trel_2 = Conv.Trellis(Conv.EncodingType.BSC)

        # Computing logMax of A, B, and Lambda Values for ForwardBackward Algorithm
        A_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        A_1[0, 0] = 0
        A_1[0, 1:] = float('-inf')
        B_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        B_1[BLOCK_LENGTH + MEMORY, 0] = 0
        B_1[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
        R_1 = np.zeros(BLOCK_LENGTH + MEMORY)
        for t in range(1, BLOCK_LENGTH + MEMORY + 1):
            A_1[t] = trel_1.updateA(A_1, coded_observation1[t-1], t, p)
        for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            B_1[t] = trel_1.updateB(B_1, coded_observation1[t], t, p)
        for t in range(BLOCK_LENGTH + MEMORY):
            R_1[t] = trel_1.updateR(A_1[t], B_1[t+1], coded_observation1[t], p)

        # BER
        R = np.zeros(BLOCK_LENGTH + MEMORY)
        R[R_1 < 0] = 0
        R[R_1 > 0] = 1
        # BER_0[block] = 1 - np.mean(R[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])
        # print(f"No Cycle BER={BER_0[block]:.6f} at p={Probability[iter]:.6f}")
        R_1pi = R_1[pi].copy()
        for c in range(CYCLES):
            A_2 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            A_2[0, 0] = 0
            A_2[0, 1:] = float('-inf')
            B_2 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            B_2[BLOCK_LENGTH + MEMORY, 0] = 0
            B_2[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
            R_2 = np.zeros(BLOCK_LENGTH + MEMORY)
            for t in range(1, BLOCK_LENGTH + MEMORY + 1):
                A_2[t] = trel_2.updateA(A_2, coded_observation2[t-1], t, p, R_1pi[t-1])
            for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
                B_2[t] = trel_2.updateB(B_2, coded_observation2[t], t, p, R_1pi[t])
            for t in range(BLOCK_LENGTH):
                R_2[t] = trel_2.updateR(A_2[t], B_2[t+1], coded_observation2[t], p)

            R_2pi = np.zeros(BLOCK_LENGTH + MEMORY)
            R_2pi[pi] = R_2.copy()

            A_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            A_1[0, 0] = 0
            A_1[0, 1:] = float('-inf')
            B_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            B_1[BLOCK_LENGTH + MEMORY, 0] = 0
            B_1[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
            R_1 = np.zeros(BLOCK_LENGTH + MEMORY)
            for t in range(1, BLOCK_LENGTH + MEMORY + 1):
                A_1[t] = trel_1.updateA(A_1, coded_observation1[t-1], t, p, R_2pi[t-1])
            for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
                B_1[t] = trel_1.updateB(B_1, coded_observation1[t], t, p, R_2pi[t])
            for t in range(BLOCK_LENGTH):
                R_1[t] = trel_1.updateR(A_1[t], B_1[t+1], coded_observation1[t], p)

            R = np.zeros(BLOCK_LENGTH + MEMORY)
            R[R_1 < 0] = 0
            R[R_1 > 0] = 1
            BER_1[block] = 1 - np.mean(R[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])

            print(f"Cycle BER={BER_1[block]:.6f} at p={Probability[iter]:.6f}")

            R_1pi = R_1[pi].copy()
    # BER_data0[iter] = np.mean(BER_0)
    BER_data1[iter] = np.mean(BER_1)

# Ploting Chart
plt.title("Turbo Code Performance (BSC)")
plt.xlabel("Probability")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Probability, BER_data1, color="blue")
# plt.plot(Probability, BER_data1, color="green")
plt.legend(["Turbo"])
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()
