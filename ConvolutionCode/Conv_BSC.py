import numpy as np
import matplotlib.pyplot as plt
import random

import Conv_Trellis as Conv

# Constants
BLOCK_LENGTH = 16000
MEMORY = 3
STATE_COUNT = (2 ** MEMORY)
ITERATIONS = 10
TRIALS = 10

# Plot Data
Probability = np.logspace(-5, -1, num=ITERATIONS)
Viterbi_BER_data = np.zeros(ITERATIONS, dtype=np.double)
FwBw_BER_data = np.zeros(ITERATIONS, dtype=np.double)


# initialize randomizer
random.seed()

for iter in range(ITERATIONS):
    # Declaring uncoded and coded matrixes
    uncoded = np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)
    #uncoded = [1, 1, 0, 0, 1, 0, 1, 0]

    coded = np.zeros((BLOCK_LENGTH + MEMORY, 2), dtype=int)

    # Randomize uncoded message
    v = 0.5
    for i in range(BLOCK_LENGTH):
        if random.random() < v:
            uncoded[i] = 1

    # encoding uncoded
    d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
    D = np.zeros((BLOCK_LENGTH + MEMORY, MEMORY), dtype=np.int16)
    for i in range(BLOCK_LENGTH):
        D[i] = d.copy()
        coded[i] = Conv.conv(uncoded[i], d, Conv.EncodingType.BSC)

    # Using the last 3 bits to reset state to [ 0, 0, 0 ]
    for i in range(BLOCK_LENGTH, BLOCK_LENGTH + MEMORY):
        D[i] = d.copy()
        resetbit = (d[1] + d[2]) % 2
        uncoded[i] = resetbit
        coded[i] = Conv.conv(resetbit, d, Conv.EncodingType.BSC)

    # introducing noise
    p = Probability[iter]
    Viterbi_BER = np.zeros(TRIALS, dtype=np.double)
    FwBw_BER = np.zeros(TRIALS, dtype=np.double)
    coded_observation = coded.copy()
    for trial in range(TRIALS):

        for i in range(BLOCK_LENGTH + MEMORY):
            for j in range(2):
                if random.random() < p:
                    coded_observation[i][j] = 0 if coded_observation[i][j] else 1
        trel = Conv.Trellis(Conv.EncodingType.BSC)
        Viterbi_uncoded_guess = np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)

        # Calculating cost branches for Viterbi Algorithm
        costMatrix = np.zeros((BLOCK_LENGTH + MEMORY, STATE_COUNT), dtype=int)
        cost = np.zeros(STATE_COUNT, dtype=int)
        for i in range(1, BLOCK_LENGTH + MEMORY):

            costMatrix[i] = trel.updateBranchCost(costMatrix[i-1], coded_observation[i-1], i)

        for i in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            Viterbi_uncoded_guess[i] = trel.viterbi_Decoder(coded_observation[i], costMatrix[i], i)

        # Computing logMax of A, B, and Lambda Values for ForwardBackward Algorithm
        A = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        A[0, 0] = 0
        A[0, 1:] = float('-inf')
        B = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        B[BLOCK_LENGTH + MEMORY, 0] = 0
        B[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
        R = np.zeros(BLOCK_LENGTH + MEMORY)
        for t in range(1, BLOCK_LENGTH + MEMORY + 1):
            A[t] = trel.updateA(A, coded_observation[t-1], t, p)
        for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            B[t] = trel.updateB(B, coded_observation[t], t, p)
        for t in range(1, BLOCK_LENGTH + MEMORY):
            R[t] = trel.updateR(A[t], B[t+1], coded_observation[t], p)

        # BER
        R[R < 0] = 0
        R[R > 0] = 1

        Viterbi_BER[trial] = 1 - np.mean(Viterbi_uncoded_guess == uncoded)
        FwBw_BER[trial] = 1 - np.mean(R == uncoded)  # misses / (BLOCK_LENGTH + MEMORY)

    Viterbi_BER_data[iter] = np.mean(Viterbi_BER)
    FwBw_BER_data[iter] = np.mean(FwBw_BER)
    print(f"Vtbi BER={Viterbi_BER_data[iter]:.6f} at p={Probability[iter]:.6f}")
    print(f"FwBw BER={FwBw_BER_data[iter]:.6f} at p={Probability[iter]:.6f}")

# Ploting Chart
plt.title("Viterbi vs ForwardBackward Algorithm (BSC)")
plt.xlabel("Probability")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Probability, Viterbi_BER_data, color="blue")
plt.plot(Probability, FwBw_BER_data, color="red")
plt.legend(["Viterbi", "ForwardBackward"])
plt.ylim(10 ** (-6), 10 ** (0))
plt.show()
