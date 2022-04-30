# Convolution Codes
The programs written for this assignment consist of a Convolution Encoder, Viterbi Decoder, and Forward Backward decoder. The two main programs to run are Conv_BSC.py and Conv_AWGN.py. Both require the Conv_Trellis.py file in the same directory to run. 
To run the BSC program, use the following command while in the directory to run them in Python3:
  python Conv_BSC.py
  or 
  python Conv_AWGN.py
Both programs will require Numpy, Scipy, matplotlib, and random installed with pip.
	The programs use words of length k=16000 with 10 words each. The BSC program was tested using probability values between 10^(-5) to 10^(-1).
 
The AWGN (soft decision) and BPSK (hard decision) channel simulations are ran on the Conv_AWGN.py file and were ran using Eb/No values from 2.7 to 6.3.
 
 
After Running these tests, I found that the average penalty for using hard decoding vs soft decoding while using AWGN on the Viterbi Decoder was about 2.5 percentage points while the average  penalty for using hard Forward/Backward Decoding was about -1.3 percentage points.

