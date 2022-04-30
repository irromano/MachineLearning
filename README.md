# Linear Density Parity Check Decoder

Linear Density Parity Check Decoder
	In this implementation of LDPC, the BSC and AWGN channels are simulated using separate python files; LDPC_BSC.py and LDPC_AWGN.py. Each file passes an n=20000 length zero-codeword into the decoder at 500 blocks. This takes approximately 12 hours each so I recommend reducing the BLOCKS constant on line 11 from 500 to 1 if you need to test it.
Each program can be ran with Python3 using the following commands:
python LDPC_BSC.py
python LDPC_BSC.py
The programs require Numpy, Scipy and matplotlib installed using the pip command to operate.
 
The program appears to run better than the predicted BER with probability values between 0.01 to 0.21
 
This used Eb /No values of 0.1 to 1.3 and also appeared to perform better than the predicted performance.
