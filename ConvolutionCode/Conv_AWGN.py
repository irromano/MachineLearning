import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

import Conv_Trellis as Conv

# Constants
BLOCK_LENGTH = 16000
MEMORY = 3
STATE_COUNT = (2 ** MEMORY)
ITERATIONS = 10
TRIALS = 1


# Plot Data
Eb_data = np.linspace(1.0, 2.8, ITERATIONS)
Viterbi_BER_data_soft = np.zeros(ITERATIONS, dtype=np.double)
Viterbi_BER_data_hard = np.zeros(ITERATIONS, dtype=np.double)
FwBw_BER_data_soft = np.zeros(ITERATIONS, dtype=np.double)
FwBw_BER_data_hard = np.zeros(ITERATIONS, dtype=np.double)

# initialize randomizer
rng = default_rng()

for iter in range(ITERATIONS):
    Viterbi_BER_soft = np.zeros(TRIALS, dtype=np.double)
    Viterbi_BER_hard = np.zeros(TRIALS, dtype=np.double)
    FwBw_BER_soft = np.zeros(TRIALS, dtype=np.double)
    FwBw_BER_hard = np.zeros(TRIALS, dtype=np.double)

    for trial in range(TRIALS):

        # Declaring uncoded and coded matrixes
        uncoded = rng.choice(2, BLOCK_LENGTH + MEMORY)  # np.zeros(BLOCK_LENGTH + MEMORY, dtype=int)
        uncoded[BLOCK_LENGTH:] = 0

        coded = np.zeros((BLOCK_LENGTH + MEMORY, 2), dtype=np.double)
        coded_guess = coded.copy()
        uncoded_guess_soft = np.zeros(BLOCK_LENGTH + MEMORY)
        uncoded_guess_hard = uncoded_guess_soft.copy()

        # encoding uncoded
        d = np.zeros(MEMORY, dtype=np.int16)     # Memory State Elements
        D = np.zeros((BLOCK_LENGTH + MEMORY, MEMORY), dtype=np.int16)
        for i in range(BLOCK_LENGTH):
            D[i] = d.copy()
            coded[i] = Conv.conv(uncoded[i], d, Conv.EncodingType.AWGN, Eb_data[iter])

        # Using the last 3 bits to reset state to [ 0, 0, 0 ]
        for i in range(BLOCK_LENGTH, BLOCK_LENGTH + MEMORY):
            D[i] = d.copy()
            resetbit = (d[1] + d[2]) % 2
            uncoded[i] = resetbit
            coded[i] = Conv.conv(resetbit, d, Conv.EncodingType.AWGN, Eb_data[iter])

        coded_observation = coded.copy()

        # introducing noise
        sigmoid = np.sqrt(1 / Eb_data[iter])
        noise = np.reshape(np.random.normal(0, sigmoid, size=(BLOCK_LENGTH + MEMORY) * 2), (BLOCK_LENGTH + MEMORY, 2))
        coded_observation += noise

        trel_soft = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter])
        trel_hard = Conv.Trellis(Conv.EncodingType.AWGN, Eb_data[iter], True)

        # Calculating cost branches for Viterbi Algorithm
        costMatrix_soft = np.zeros((BLOCK_LENGTH + MEMORY, STATE_COUNT), dtype=np.double)
        cost_soft = np.zeros(STATE_COUNT, dtype=np.double)
        costMatrix_hard = np.zeros((BLOCK_LENGTH + MEMORY, STATE_COUNT), dtype=np.double)
        cost_hard = np.zeros(STATE_COUNT, dtype=np.double)
        for i in range(1, BLOCK_LENGTH + MEMORY):
            costMatrix_soft[i] = trel_soft.updateBranchCost(costMatrix_soft[i-1], coded_observation[i-1], i)
            costMatrix_hard[i] = trel_hard.updateBranchCost(costMatrix_hard[i-1], coded_observation[i-1], i)

        for i in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            uncoded_guess_soft[i] = trel_soft.viterbi_Decoder(coded_observation[i], costMatrix_soft[i], i)
            uncoded_guess_hard[i] = trel_hard.viterbi_Decoder(coded_observation[i], costMatrix_hard[i], i)

        # Computing logMax of A, B, and Lambda Values for ForwardBackward Algorithm
        A_soft = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        A_soft[0, 0] = 0
        A_soft[0, 1:] = float('-inf')
        A_hard = A_soft.copy()
        B_soft = np.zeros((BLOCK_LENGTH + MEMORY + 1, STATE_COUNT))
        B_soft[BLOCK_LENGTH + MEMORY, 0] = 0
        B_soft[BLOCK_LENGTH + MEMORY, 1:] = float('-inf')
        B_hard = B_soft.copy()
        R_soft = np.zeros(BLOCK_LENGTH + MEMORY)
        R_hard = R_soft.copy()

        for t in range(1, BLOCK_LENGTH + MEMORY + 1):
            A_soft[t] = trel_soft.updateA(A_soft, coded_observation[t-1], t, Eb_data[iter])
            A_hard[t] = trel_hard.updateA(A_hard, coded_observation[t-1], t, Eb_data[iter])
        for t in range(BLOCK_LENGTH + MEMORY - 1, -1, -1):
            B_soft[t] = trel_soft.updateB(B_soft, coded_observation[t], t, Eb_data[iter])
            B_hard[t] = trel_hard.updateB(B_hard, coded_observation[t], t, Eb_data[iter])
        for t in range(BLOCK_LENGTH + MEMORY):
            R_soft[t] = trel_soft.updateR(A_soft[t], B_soft[t+1], coded_observation[t], Eb_data[iter])
            R_hard[t] = trel_hard.updateR(A_hard[t], B_hard[t+1], coded_observation[t], Eb_data[iter])

        # BER
        R_soft[R_soft < 0] = 0
        R_soft[R_soft > 0] = 1
        R_hard[R_hard < 0] = 0
        R_hard[R_hard > 0] = 1

        # BER calculation
        Viterbi_BER_soft[trial] = 1 - np.mean(uncoded_guess_soft == uncoded)
        Viterbi_BER_hard[trial] = 1 - np.mean(uncoded_guess_hard == uncoded)
        FwBw_BER_soft[trial] = 1 - np.mean(R_soft == uncoded)
        FwBw_BER_hard[trial] = 1 - np.mean(R_hard == uncoded)

    Viterbi_BER_data_soft[iter] = np.mean(Viterbi_BER_soft)
    Viterbi_BER_data_hard[iter] = np.mean(Viterbi_BER_hard)
    FwBw_BER_data_soft[iter] = np.mean(FwBw_BER_soft)
    FwBw_BER_data_hard[iter] = np.mean(FwBw_BER_hard)
    print(f"Eb/No = {Eb_data[iter]:.6f} BER soft = {Viterbi_BER_data_soft[iter]:.6f}")
    print(f"Eb/No = {Eb_data[iter]:.6f} BER hard = {Viterbi_BER_data_hard[iter]:.6f}")
    print(f"Eb/No = {Eb_data[iter]:.6f} FwBw BER soft = {FwBw_BER_data_soft[iter]:.6f}")
    print(f"Eb/No = {Eb_data[iter]:.6f} FwBw BER hard = {FwBw_BER_data_hard[iter]:.6f}")

print(f"The Average Viterbi Soft Error was {np.mean(Viterbi_BER_data_soft)}")
print(f"The Average Viterbi Hard Error was {np.mean(Viterbi_BER_data_hard)}")
print(f"The Average FwBw Soft Error was {np.mean(FwBw_BER_data_soft)}")
print(f"The Average FwBw Hard Error was {np.mean(FwBw_BER_data_hard)}")
print(f"The Average Viterbi Hard penality was {np.mean(Viterbi_BER_data_hard) - np.mean(Viterbi_BER_data_soft)}")
print(f"The Average FwBw Hard penality was {np.mean(FwBw_BER_data_hard) - np.mean(FwBw_BER_data_soft)}")

# Ploting Chart
plt.title("Viterbi vs ForwardBackward Algorithm (AWGN)")
plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Eb_data, Viterbi_BER_data_soft, color="red", label="Viterbi Soft")
plt.plot(Eb_data, FwBw_BER_data_soft, color="green", label="FwBw Soft")
plt.legend(["Viterbi Soft", "FwBw soft"])
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()

plt.title("Viterbi vs ForwardBackward Algorithm (AWGN/BPSK)")
plt.xlabel("Eb/No")
plt.ylabel("BER")
plt.yscale("log")
plt.plot(Eb_data, Viterbi_BER_data_soft, color="red", label="Viterbi Soft")
plt.plot(Eb_data, Viterbi_BER_data_hard, color="blue", label="Viterbi Hard")
plt.plot(Eb_data, FwBw_BER_data_soft, color="green", label="FwBw Soft")
plt.plot(Eb_data, FwBw_BER_data_hard, color="purple", label="FwBw hard")
plt.legend(["Viterbi Soft", "Viterbi Hard", "FwBw Soft", "FwBw Hard"])
plt.ylim(10 ** (-6), 10 ** (-1))
plt.show()
