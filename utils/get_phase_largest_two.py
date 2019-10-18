import math
import numpy as np
from utils.random import *
from utils.relative_phase_noise import *
import itertools # used for combinations and permutations
from utils.get_phase_bipartite import *
from utils.Mixon_comparison_2 import *

def overall_largest_two(magnitude_array): # this function finds the index corresponding to the largest magnitude for both the even ones and odd ones.  For whichever is bigger, it also finds the second largest for that half of the vector
    (even_ones_index, even_ones_maximum, odd_ones_index, odd_ones_maximum) = find_max_even_and_odd(magnitude_array)
    (max_index1, max_mag1, second_index, second_mag, parity) = overall_max(even_ones_index, even_ones_maximum, odd_ones_index, odd_ones_maximum)
    
    m = int(math.log(len(magnitude_array),2))
    (even_ones_indices, odd_ones_indices) = get_indices(m)
    if parity:
        odd_ones_indices.remove(max_index1)
        
        max_index2 = -1
        max_mag2 = -1
        for index in odd_ones_indices:             # find max magnitude and index for even number of ones
            if (magnitude_array[index] > max_mag2):
                max_mag2 = magnitude_array[index]
                max_index2 = index
    else:
        even_ones_indices.remove(max_index1)
        
        max_index2 = -1
        max_mag2 = -1
        for index in even_ones_indices:             # find max magnitude and index for even number of ones
            if (magnitude_array[index] > max_mag2):
                max_mag2 = magnitude_array[index]
                max_index2 = index
    
    return (max_index1, max_mag1, max_index2, max_mag2, second_index, second_mag, parity)

#TESTING 10/2/19, working
# print(overall_largest_two([10,20,30,40,50,60,70,80]))


def get_phase_largest_two(x,deviation): #This function takes a vector of length 2**m for some positive integer m and uses intensity and relative phase measurements to return that vector up to a global phase factor
    if (len(x) == 1):                    # if vector is of length one, just return an intensity measurement of the entry
        return np.array([intensity_measurement_bipartite(x[0],deviation)])
    
    if (len(x) == 2):
        print("ERROR get_phase_bipartite_two_largest: length 2 for signal will not work with this method")
        return
    
    (m, x, power_of_two, old_length) = check_if_power_of_two(x) # extends the vector x if necessary to have lenght a power of 2
    
    x_hat1 = np.zeros(len(x), complex)     # the return of this function is the estimate x_hat1
    x_hat2 = np.zeros(len(x), complex)     # the return of this function is the estimate x_hat2

    magnitude_array = magnitudes(x, deviation, old_length) # take N measurements to get the magnitudes of each entry
    (max_index1, max_mag1, max_index2, max_mag2, second_index, second_mag, parity) = overall_largest_two(magnitude_array)
    
    x_hat1[max_index1] = max_mag1       # puts the estimate of the starting vertex in x_hat
    x_hat2[max_index2] = max_mag2
    
    ### x_hat1
    if parity:                       # valid_indices1 are the indices adjacent to max_index and valid_indices2 are the indices adjacent to second_index
        (valid_indices1, valid_indices2) = get_indices(m)
    else:
        (valid_indices2, valid_indices1) = get_indices(m)
    valid_indices2.remove(max_index1) # we don't want to get another estimate for max_index
    
    for index in valid_indices1:     # starting from the largest magnitude vertex, get an estimate for the phase of all entries on the opposite side of the graph with 4 * N/2 measurements, i.e. N/2 relative phase measurements
        rel_phase = relative_phase_noise( (x[max_index1], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat1[index] = np.multiply(rel_phase, (1.0/ x_hat1[max_index1].conjugate()))
        
    for index in valid_indices2:     # now from the largest magnitude vertex of the opposite side of the graph, get an estimate for the phase of all entries except the starting entry on the starting side of the graph with 4 * (N/2 - 1) measurements, i.e. N/2 - 1 relative phase measurements
        rel_phase = relative_phase_noise( (x[second_index], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat1[index] = np.multiply(rel_phase, (1.0/ x_hat1[second_index].conjugate()))
    
    ### x_hat2
    if parity:                       # valid_indices1 are the indices adjacent to max_index and valid_indices2 are the indices adjacent to second_index
        (valid_indices1, valid_indices2) = get_indices(m)
    else:
        (valid_indices2, valid_indices1) = get_indices(m)
    valid_indices2.remove(max_index2) # we don't want to get another estimate for max_index
    
    for index in valid_indices1:     # starting from the largest magnitude vertex, get an estimate for the phase of all entries on the opposite side of the graph with 4 * N/2 measurements, i.e. N/2 relative phase measurements
        rel_phase = relative_phase_noise( (x[max_index2], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat2[index] = np.multiply(rel_phase, (1.0/ x_hat2[max_index2].conjugate()))
        
    for index in valid_indices2:     # now from the largest magnitude vertex of the opposite side of the graph, get an estimate for the phase of all entries except the starting entry on the starting side of the graph with 4 * (N/2 - 1) measurements, i.e. N/2 - 1 relative phase measurements
        rel_phase = relative_phase_noise( (x[second_index], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat2[index] = np.multiply(rel_phase, (1.0/ x_hat2[second_index].conjugate()))
    
    
    relative_phase_factor = (x_hat1[max_index1]/x_hat2[max_index1])/np.sqrt((x_hat1[max_index1]/x_hat2[max_index1])*(x_hat1[max_index1]/x_hat2[max_index1]).conjugate())
#     print("rel phase factor", relative_phase_factor, np.absolute(relative_phase_factor))
    
    
    x_hat = (0.5)*(x_hat1 + np.multiply(relative_phase_factor, x_hat2))
#     print("x_hat1", x_hat1)
#     print("x_hat2", np.multiply(relative_phase_factor,x_hat2))
#     print()
#     print("x_hat", x_hat)
#     print("------------")
    
    global_phase_factor = (x[max_index1]/x_hat[max_index1])/np.sqrt((x[max_index1]/x_hat[max_index1])*(x[max_index1]/x_hat[max_index1]).conjugate())

    if not power_of_two:             # if the length of the signal is not a power of 2, then there are junk entries at the end of x_hat that need to be removed
        temp = x_hat
        x_hat = np.zeros(old_length, complex)
        for i in range(old_length):
            x_hat[i] = temp[i]
    
    return x_hat, global_phase_factor


def test_bipartite_v_largest_two(file_name): # given a certain number of trials per value of N, this function evaluates the average MSE (relative error not squared relative error) of getPhaseNoiseBipartite in a way that is comparable to the paper Phase Retrieval with Polarization by Mixon
    signal_vectors, noise_vectors = get_vectors_and_noise_from_Mixon(file_name)
    
    average_RE_bipartite = []               # records the average MSE for each value of N
    average_RE_largest_two = []               # records the average MSE for each value of N

    for i in tqdm(range(len(signal_vectors))):
#         print(i)
        x = signal_vectors[i]   # create the signal of length N with standard deviation 1/sqrt(N)
        
        N = len(x)
        m = int(math.log(N,2))     # get m = log(N) always rounding down
        
        noise_deviation  = 0.4/np.sqrt(N)          # nu   ~ N(0, (0.4)^2 /N  )
        
        ### bipartite
        x_hat_array = []
        for j in range(2):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
            (x_hat_bipartite, gpf) = get_phase_bipartite(x, noise_deviation) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
            x_hat_array.append(x_hat_bipartite*gpf)
        
        x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
        average_RE_bipartite.append(relative_error(x_hat, x))
        
        ### largest two
        (x_hat, gpf) = get_phase_largest_two(x, noise_deviation) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4        
        average_RE_largest_two.append(relative_error(x_hat*gpf, x))
        
        # end for loop
    
    x_axis = []
    for signal in signal_vectors: 
        x_axis.append(len(signal))
        
    plt.plot(x_axis, average_RE_bipartite, color = 'b', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    plt.plot(x_axis, average_RE_largest_two, color = 'g', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    #CHANGE
    #     title = "Relative Error versus N Bipartite vs. Largest Two"
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("Relative Error")
#     upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
#     plt.axis([0, 130, 0, 0.05])            # set axes
    plt.show
#     plt.savefig('graphs/Ben_.png')
    return average_RE_bipartite, average_RE_largest_two