import numpy as np
from collections import deque

def get_vectors_and_noise_from_Mixon(file_name):    # this function uses the name of a text file "file_name" to access a file of Matlab outputs that allow for direct comparison of our algorithm with Mixon's
    file = open(file_name,"r") 
    signals_array = []         # the return, a signal x for each value of N
    noises_array  = []         # the return, a noise queue for each signal x
    
    while(True):                       # repeat until the end of a file
        read = file.readline().strip()
        if (read=="END"):              # END denotes the end of a file
            break
        elif (read=="SIG"):
            pass
        else:
            print("ERROR: get_vectors_and_noise_from_Mixon: read is:", read, "when it should be SIG")
            break
        
        N = int(file.readline())       # N is the length of the vector x
        signals_reals = np.zeros(N)
        signals_imag  = np.zeros(N)
        signals_vec   = np.zeros(N, complex)
        
        for i in range(N):             # gets each entry of x one line at a time
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            signals_reals[i] = val* (10**exp) #first gets the real parts
        
        for i in range(N):
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            signals_imag[i] = val* (10**exp) #then gets the imaginary parts
            
        for i in range(N):                   #combines them correctly before appending to array of signals of each length
            signals_vec[i] = (signals_reals[i] + 1j*signals_imag[i])
        
        signals_array.append(signals_vec)
#         signals_array = np.append(signals_array, [signals_vec])
    
        read = file.readline().strip()
        if (read=="NOISE"):
            pass
        else:
            print("ERROR: get_vectors_and_noise_from_Mixon: read is:", read, "when it should be NOISE")
            break
        
        l = int(file.readline())       # l is the length of the noise vector
        noises_reals  = np.zeros(l)
        noises_imag   = np.zeros(l)
        noises_queue  = deque()        
        
        for i in range(l):             # gets each entry of x one line at a time
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            noises_reals[i] = val* (10**exp) #first gets the real parts
        
        for i in range(l):             # gets each entry of x one line at a time
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            noises_imag[i] = val* (10**exp) #then gets the imaginary parts
            
        for i in range(l):
            noises_queue.append(noises_reals[i] + 1j*noises_imag[i])
        
        noises_array.append(noises_queue)
        
    file.close()
    return signals_array, noises_array

# check, working 9/21/19
# signals_list, noises_list = get_vectors_and_noise_from_Mixon("data/signal_and_noise_128.txt")
# print(len(signals_list))
# # for v in signals_list:
# #     print(v)
# #     print()
# print("=============================================================")
# print(len(noises_list))
# M = 8
# for n in noises_list:
#     print(M, len(n), len(n)/M)
#     M = M+4;
# #     print()