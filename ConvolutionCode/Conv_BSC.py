import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

import Conv_Trellis as Conv

# Constants
BLOCK_LENGTH = 16000
MEMORY = 3
STATE_COUNT = (2 ** MEMORY)
ITERATIONS = 10
TRIALS = 10

# Plot Data
Probability = np.linspace(0.001, 0.091, num=ITERATIONS)
Viterbi_BER_data = np.zeros(ITERATIONS, dtype=np.double)
FwBw_BER_data = np.zeros(ITERATIONS, dtype=np.double)


# initialize randomizer
rng = default_rng()

for iter in range(ITERATIONS):
    Viterbi_BER = np.zeros(TRIALS, dtype=np.double)
    FwBw_BER = np.zeros(TRIALS, dtype=np.double)
    for trial in range(TRIALS):
        # Declaring uncoded and coded matrixes
        uncoded = rng.choice(2, BLOCK_LENGTH + MEMORY)  # np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)
        uncoded[BLOCK_LENGTH:] = 0

        coded = np.zeros((BLOCK_LENGTH + MEMORY, 2), dtype=int)

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
        coded_observation = coded.copy()
        error = rng.choice(2, (BLOCK_LENGTH + MEMORY, 2), p=[1-p, p])
        coded_observation = (coded_observation + error) % 2

        trel_Viterbi = Conv.Trellis(Conv.EncodingType.BSC)
        trel_FwBw = Conv.Trellis(Conv.EncodingType.BSC)
        Viterbi_uncoded_guess = np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)

        # Calculating cost branches for Viterbi Algorithm
        costMatrix = np.zeros((BLOCK_LENGTH + MEMORY, STATE_COUNT), dtype=int)
        cost = np.zeros(STATE_COUNT, dtype=int)
        for i in range(1, BLOCK_LENGTH + MEMORY):

            costMatrix[i] = trel_Viterbi.updateBranchCost(costMatrix[i-1], coded_observation[i-1], i)

        for i in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            Viterbi_uncoded_guess[i] = trel_Viterbi.viterbi_Decoder(coded_observation[i], costMatrix[i], i)

        # Computing logMax of A, B, and Lambda Values for ForwardBackward Algorithm
        A = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        A[0, 0] = 0
        A[0, 1:] = float('-inf')
        B = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        B[BLOCK_LENGTH + MEMORY, 0] = 0
        B[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
        R = np.zeros(BLOCK_LENGTH + MEMORY)
        for t in range(1, BLOCK_LENGTH + MEMORY + 1):
            A[t] = trel_FwBw.updateA(A, coded_observation[t-1], t, p)
        for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            B[t] = trel_FwBw.updateB(B, coded_observation[t], t, p)
        for t in range(BLOCK_LENGTH + MEMORY):
            R[t] = trel_FwBw.updateR(A[t], B[t+1], coded_observation[t], p)

        # BER
        R[R < 0] = 0
        R[R > 0] = 1

        Viterbi_BER[trial] = 1 - np.mean(Viterbi_uncoded_guess == uncoded)
        FwBw_BER[trial] = 1 - np.mean(R == uncoded)

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
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()
