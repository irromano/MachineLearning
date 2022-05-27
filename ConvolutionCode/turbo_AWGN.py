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
ITERATIONS = 7
BLOCKS = 4
CYCLES = 1

Eb_data = np.linspace(0.9, 2.1, ITERATIONS)
BER_data0_soft = np.zeros(ITERATIONS, dtype=np.double)
BER_data0_hard = np.zeros(ITERATIONS, dtype=np.double)
BER_data_soft = np.zeros(ITERATIONS, dtype=np.double)
BER_data_hard = np.zeros(ITERATIONS, dtype=np.double)

# initialize randomizer
rng = default_rng()

for iter in range(ITERATIONS):
    BER_0_soft = np.zeros(BLOCKS, dtype=np.double)
    BER_0_hard = BER_0_soft.copy()
    BER_soft = np.zeros(BLOCKS, dtype=np.double)
    BER_hard = np.zeros(BLOCKS, dtype=np.double)
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
            coded1[i] = Conv.conv(uncoded1[i], d, Conv.EncodingType.AWGN, Eb_data[iter])

        # encoding permutation
        d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
        D = np.zeros((BLOCK_LENGTH + MEMORY, MEMORY), dtype=np.int16)
        for i in range(BLOCK_LENGTH):
            D[i] = d.copy()
            coded2[i] = Conv.conv(uncoded2[i], d, Conv.EncodingType.AWGN, Eb_data[iter])

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
        coded_observation1 = coded1.copy()
        sigmoid = np.sqrt(1 / Eb_data[iter])
        error1 = np.reshape(np.random.normal(0, sigmoid, size=(BLOCK_LENGTH + MEMORY) * 2), (BLOCK_LENGTH + MEMORY, 2))
        error2 = np.reshape(np.random.normal(0, sigmoid, size=(BLOCK_LENGTH + MEMORY) * 2), (BLOCK_LENGTH + MEMORY, 2))
        coded_observation1 += error1
        coded_observation2 += error2

        # P of flip for hard desicion
        p = 1 - stats.norm.cdf(Eb_data[iter], scale=sigmoid)

        trel_soft_1 = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter])
        trel_soft_2 = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter])
        trel_hard_1 = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter], True)
        trel_hard_2 = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter], True)

        # Computing logMax of A, B, and Lambda Values for ForwardBackward Algorithm
        A_soft_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        A_soft_1[0, 0] = 0
        A_soft_1[0, 1:] = float('-inf')
        A_hard_1 = A_soft_1.copy()
        B_soft_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        B_soft_1[BLOCK_LENGTH + MEMORY, 0] = 0
        B_soft_1[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
        B_hard_1 = B_soft_1.copy()
        R_soft_1 = np.zeros(BLOCK_LENGTH + MEMORY)
        R_hard_1 = R_soft_1.copy()
        for t in range(1, BLOCK_LENGTH + MEMORY + 1):
            A_soft_1[t] = trel_soft_1.updateA(A_soft_1, coded_observation1[t-1], t, Eb_data[iter])
            A_hard_1[t] = trel_hard_1.updateA(A_hard_1, coded_observation1[t-1], t, p)
        for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            B_soft_1[t] = trel_soft_1.updateB(B_soft_1, coded_observation1[t], t, Eb_data[iter])
            B_hard_1[t] = trel_hard_1.updateB(B_hard_1, coded_observation1[t], t, p)
        for t in range(BLOCK_LENGTH + MEMORY):
            R_soft_1[t] = trel_soft_1.updateR(A_soft_1[t], B_soft_1[t+1], coded_observation1[t], Eb_data[iter])
            R_hard_1[t] = trel_hard_1.updateR(A_hard_1[t], B_hard_1[t+1], coded_observation1[t], p)

        # BER
        R_soft = np.zeros(BLOCK_LENGTH + MEMORY)
        R_hard = R_soft.copy()
        R_soft[R_soft_1 < 0] = 0
        R_soft[R_soft_1 > 0] = 1
        R_hard[R_hard_1 < 0] = 0
        R_hard[R_hard_1 > 0] = 1

        # BER_0_soft[block] = 1 - np.mean(R_soft[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])
        # BER_0_hard[block] = 1 - np.mean(R_hard[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])
        # print(f"Soft Cycle BER={BER_0_soft[block]:.6f} at EB/No={Eb_data[iter]:.6f}")
        # print(f"Hard Cycle BER={BER_0_hard[block]:.6f} at EB/No={Eb_data[iter]:.6f}")
        R_soft_1pi = R_soft_1[pi].copy()
        R_hard_1pi = R_hard_1[pi].copy()
        for c in range(CYCLES):
            A_soft_2 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            A_soft_2[0, 0] = 0
            A_soft_2[0, 1:] = float('-inf')
            A_hard_2 = A_soft_2.copy()
            B_soft_2 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            B_soft_2[BLOCK_LENGTH + MEMORY, 0] = 0
            B_soft_2[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
            B_hard_2 = B_soft_2.copy()
            R_soft_2 = np.zeros(BLOCK_LENGTH + MEMORY)
            R_hard_2 = R_soft_2.copy()
            for t in range(1, BLOCK_LENGTH + MEMORY + 1):
                A_soft_2[t] = trel_soft_2.updateA(A_soft_2, coded_observation2[t-1], t, Eb_data[iter], R_soft_1pi[t-1])
                A_hard_2[t] = trel_hard_2.updateA(A_hard_2, coded_observation2[t-1], t, p, R_hard_1pi[t-1])
            for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
                B_soft_2[t] = trel_soft_2.updateB(B_soft_2, coded_observation2[t], t, Eb_data[iter], R_soft_1pi[t])
                B_hard_2[t] = trel_hard_2.updateB(B_hard_2, coded_observation2[t], t, p, R_hard_1pi[t])
            for t in range(BLOCK_LENGTH):
                R_soft_2[t] = trel_soft_2.updateR(A_soft_2[t], B_soft_2[t+1], coded_observation2[t], Eb_data[iter])
                R_hard_2[t] = trel_hard_2.updateR(A_hard_2[t], B_hard_2[t+1], coded_observation2[t], p)

            R_soft_2pi = np.zeros(BLOCK_LENGTH + MEMORY)
            R_hard_2pi = R_soft_2pi.copy()
            R_soft_2pi[pi] = R_soft_2.copy()
            R_hard_2pi[pi] = R_hard_2.copy()

            A_soft_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            A_soft_1[0, 0] = 0
            A_soft_1[0, 1:] = float('-inf')
            A_hard_1 = A_soft_1.copy()
            B_soft_1 = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
            B_soft_1[BLOCK_LENGTH + MEMORY, 0] = 0
            B_soft_1[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
            B_hard_1 = B_soft_1.copy()
            R_soft_1 = np.zeros(BLOCK_LENGTH + MEMORY)
            R_hard_1 = R_soft_1.copy()
            for t in range(1, BLOCK_LENGTH + MEMORY + 1):
                A_soft_1[t] = trel_soft_1.updateA(A_soft_1, coded_observation1[t-1], t, Eb_data[iter], R_soft_2pi[t-1])
                A_hard_1[t] = trel_hard_1.updateA(A_hard_1, coded_observation1[t-1], t, p, R_hard_2pi[t-1])
            for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
                B_soft_1[t] = trel_soft_1.updateB(B_soft_1, coded_observation1[t], t, Eb_data[iter], R_soft_2pi[t])
                B_hard_1[t] = trel_hard_1.updateB(B_hard_1, coded_observation1[t], t, p, R_hard_2pi[t])
            for t in range(BLOCK_LENGTH):
                R_soft_1[t] = trel_soft_1.updateR(A_soft_1[t], B_soft_1[t+1], coded_observation1[t], Eb_data[iter])
                R_hard_1[t] = trel_hard_1.updateR(A_hard_1[t], B_hard_1[t+1], coded_observation1[t], p)

            R_soft = np.zeros(BLOCK_LENGTH + MEMORY)
            R_hard = R_soft.copy()
            R_soft[R_soft_1 < 0] = 0
            R_soft[R_soft_1 > 0] = 1
            R_hard[R_hard_1 < 0] = 0
            R_hard[R_hard_1 > 0] = 1
            BER_soft[block] = 1 - np.mean(R_soft[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])
            BER_hard[block] = 1 - np.mean(R_hard[:BLOCK_LENGTH] == uncoded1[:BLOCK_LENGTH])

            print(f"Soft Cycle1 BER={BER_soft[block]:.6f} at EB/No={Eb_data[iter]:.6f}")
            print(f"Hard Cycle1 BER={BER_hard[block]:.6f} at EB/No={Eb_data[iter]:.6f}")

            R_soft_1pi = R_soft_1[pi].copy()
            R_hard_1pi = R_hard_1[pi].copy()
    # BER_data0_soft[iter] = np.mean(BER_0_soft)
    # BER_data0_hard[iter] = np.mean(BER_0_hard)
    BER_data_soft[iter] = np.mean(BER_soft)
    BER_data_hard[iter] = np.mean(BER_hard)

print(f"The Average Soft Error was {np.mean(BER_data_soft)}")
print(f"The Average Hard Error was {np.mean(BER_data_hard)}")
print(f"The Average Hard penality was {np.mean(BER_data_hard) - np.mean(BER_data_soft)}")

# Ploting Chart
plt.title("Turbo Code Performance (AWGN)")
plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
# plt.plot(Eb_data, BER_data0_soft, color="blue")
# plt.plot(Eb_data, BER_data0_hard, color="red")
plt.plot(Eb_data, BER_data_soft, color="green")
plt.legend(["Turbo"])
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()

# Ploting Chart
plt.title("Turbo Code Performance (AWGN/BPSK)")
plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
# plt.plot(Eb_data, BER_data0_soft, color="blue")
# plt.plot(Eb_data, BER_data0_hard, color="red")
plt.plot(Eb_data, BER_data_soft, color="green")
plt.plot(Eb_data, BER_data_hard, color="purple")
plt.legend(["Turbo Soft", "Turbo Hard"])
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()
