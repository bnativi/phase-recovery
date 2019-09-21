import numpy as np
import math
from tqdm import tqdm_notebook as tqdm

from utils.random import *
from utils.test2 import *

# Tell Python to include plots as embedded graphics.
# %matplotlib inline

# Import plotting, numpy, and library commands
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

def test_bipartite_v_Mixon_1(trials): # given a certain number of trials per value of N, this function evaluates the average MSE (relative error not squared relative error) of getPhaseNoiseBipartite in a way that is comparable to the paper Phase Retrieval with Polarization by Mixon
    N = 8                          # start at 8 and increase in increments of 4 to 128 for a total of 31 values of N tested
    average_MSE = []               # records the average MSE for each value of N
#     average_MSE_ASMT = []
    while(N <= 128):
        print(N)
        m = int(math.log(N,2))     # get m = log(N) always rounding down
        
        MSE_array = []             # records the MSE for each trial of this fixed value of N
#         MSE_array_ASMT = []

        signal_deviation = 1.0/np.sqrt(np.sqrt(N)) # x[i] ~ N(0, 1/(N^{1/2}) ) 
        noise_deviation  = 1.0/np.sqrt(N)          # nu   ~ N(0, (0.4)^2 /N  )
    
        for i in tqdm(range(trials)):
            x = generate_x_normal(signal_deviation)   # create the signal of length N with standard deviation 1/sqrt(N)
            
            x_hat_array = []
#             x_hat_array_ASMT = []
            for j in range(m):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
                (x_hat_bipartite, gpf) = get_phase_bipartite(x, noise_deviation ) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
                x_hat_array.append(x_hat_bipartite*gpf)
                
#                 (x_hat_bipartite_ASMT, gpf_ASMT) = getPhaseNoiseBipartiteASMT(x, noise_deviation ) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
#                 x_hat_array_ASMT.append(x_hat_bipartite_ASMT*gpf_ASMT)
            
            x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry 
            MSE_array.append(relative_error(x_hat, x))
            
#             x_hat_ASMT = np.average(x_hat_array_ASMT, axis=0) # average the log(N) estimates for each entry 
#             MSE_array_ASMT.append(np.linalg.norm(x_hat_ASMT - x)/np.linalg.norm(x))
            #end for loop
            
        average_MSE.append(np.average(MSE_array))
#         average_MSE_ASMT.append(np.average(MSE_array_ASMT))
        
        N+= 4
        if (N==20 or N==36 or N == 68):             # used for efficiency purposes, do less trials when each trial will take longer for longer vectors
            trials = int(trials/2.0)
        # end while loop
    
    x_axis = []
    N = 8
    
    while(N<=128): 
        x_axis.append(N)
        N += 4
        
    plt.plot(x_axis, average_MSE, color = 'b', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
#     plt.plot(x_axis, average_MSE_ASMT, color = 'g', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    title = "Relative Error versus N with Mixon parameters"
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("Relative Error")
#     upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
#     plt.axis([0, 140, 0, upperY])            # set axes
    plt.show
    return average_MSE
#     return (average_MSE, average_MSE_ASMT)


#===========================================================================================================================
def get_vectors_from_Mixon(file_name):    # this function uses the name of a text file "file_name" to access a file of Matlab outputs that allow for direct comparison of our algorithm with Mixon's
    file = open(file_name,"r") 
    signals_array = []                 # the return
    
    while(True):                       # repeat until the end of a file
        read = file.readline().strip() # READ
        if (read=="END"):              # END denotes the end of a file
            break
        elif (read=="SIG"):
            pass
        else:
            print("ERROR: getVectorsAndNoiseFromMixon: read is:", read, "when it should be SIG")
            break
        
        N = int(file.readline()) # READ                  # N is the length of the vector x
        real_vector = np.zeros(N)
        complex_vector = np.zeros(N)
        
        for i in range(N):             # gets each entry of x one line at a time
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            real_vector[i] = val* (10**exp)
        
        for i in range(N):             # gets each entry of x one line at a time
            (read1, read2) = file.readline().split("e")  # numbers are written as  "valeexp" i.e. value then the letter e then the exponent
            val = float(read1)
            exp = int(read2)
            complex_vector[i] = val* (10**exp)
        
        signals_array.append(real_vector + 1j * complex_vector)
    
    file.close()
    return signals_array

# print(getVectorsFromMixon("Mixon_vectors_small.txt"))

#===========================================================================================================================
def test_bipartite_v_Mixon_2(file_name): # given a certain number of trials per value of N, this function evaluates the average MSE (relative error not squared relative error) of getPhaseNoiseBipartite in a way that is comparable to the paper Phase Retrieval with Polarization by Mixon
    signal_vectors = get_vectors_from_Mixon(file_name)

    average_MSE = []               # records the average MSE for each value of N
    for signal in tqdm(signal_vectors):
        start = time.time()
        N = len(signal)
        print(N)
        m = int(math.log(N,2))     # get m = log(N) always rounding down
        
        MSE_array = []             # records the MSE for each trial of this fixed value of N

        signal_deviation = 1.0/np.sqrt(np.sqrt(N)) # x[i] ~ N(0, 1/(N^{1/2}) ) 
        noise_deviation  = 1.0/np.sqrt(N)          # nu   ~ N(0, (0.4)^2 /N  )
    
        x = signal   # create the signal of length N with standard deviation 1/sqrt(N)

        x_hat_array = []
        for j in range(m):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
            (x_hat_bipartite, gpf) = get_phase_bipartite(x, noise_deviation ) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
            x_hat_array.append(x_hat_bipartite*gpf)

        x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry 
        MSE_array.append(relative_error(x_hat, x))
            
        average_MSE.append(np.average(MSE_array))
        
        end = time.time()
        print(end-start)
        # end while loop
    
    x_axis = []
    for signal in signal_vectors: 
        x_axis.append(len(signal))
        
    plt.plot(x_axis, average_MSE, color = 'b', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    title = "Relative Error versus N with Mixon parameters"
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("Relative Error")
#     upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
#     plt.axis([0, 140, 0, upperY])            # set axes
    plt.show
    return average_MSE