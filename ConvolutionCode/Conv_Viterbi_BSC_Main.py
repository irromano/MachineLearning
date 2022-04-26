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
BER_data = np.zeros(ITERATIONS, dtype=np.double)


# initialize randomizer
random.seed()

for iter in range(ITERATIONS):
    # Declaring uncoded and coded matrixes
    uncoded = np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)
    #uncoded = [1, 1, 0, 0, 1, 0, 1, 0]

    coded = np.zeros((BLOCK_LENGTH + MEMORY, 2), dtype=int)
    coded_guess = coded.copy()
    uncoded_guess = uncoded.copy()

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
    BER = np.zeros(TRIALS, dtype=np.double)
    coded_observation = coded.copy()
    for trial in range(TRIALS):

        for i in range(BLOCK_LENGTH + MEMORY):
            for j in range(2):
                if random.random() < Probability[iter]:
                    coded_observation[i][j] = 0 if coded_observation[i][j] else 1
        trel = Conv.Trellis(Conv.EncodingType.BSC)
        costMatrix = np.zeros((BLOCK_LENGTH + MEMORY, STATE_COUNT), dtype=int)
        cost = np.zeros(STATE_COUNT, dtype=int)
        for i in range(1, BLOCK_LENGTH + MEMORY):

            costMatrix[i] = trel.updateBranchCost(costMatrix[i-1], coded_observation[i-1], i)

        for i in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            uncoded_guess[i] = trel.viterbi_Decoder(coded_observation[i], costMatrix[i], i)

        # BER
        misses = 0
        for i in range(BLOCK_LENGTH + MEMORY):
            if uncoded_guess[i] != uncoded[i]:
                misses += 1

        BER[trial] = misses / (BLOCK_LENGTH + MEMORY)

    BER_data[iter] = np.mean(BER)
    print(f"Iter {iter} BER={np.mean(BER):.6f} at p={Probability[iter]:.6f}")

# Ploting Chart
plt.title("Viterbi BER for probability")
plt.xlabel("Probability")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Probability, BER_data, color="red")
plt.ylim(10 ** (-6), 10 ** (0))
plt.show()