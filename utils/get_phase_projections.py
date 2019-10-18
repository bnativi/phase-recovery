import math
import numpy as np
from utils.random import *
from utils.relative_phase_noise import *
import itertools # used for combinations and permutations
from utils.get_phase_bipartite import *
from utils.Mixon_comparison_2 import *

def largest_two_both_sides(magnitude_array): # this function finds the index corresponding to the largest magnitude for both the even ones and odd ones.  For whichever is bigger, it also finds the second largest for that half of the vector
    (even_ones_index, even_ones_maximum, odd_ones_index, odd_ones_maximum) = find_max_even_and_odd(magnitude_array)
    (max_index1, max_mag1, second_index1, second_mag1, parity) = overall_max(even_ones_index, even_ones_maximum, odd_ones_index, odd_ones_maximum)
    
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
        
        even_ones_indices.remove(second_index1)
        
        second_index2 = -1
        second_mag2 = -1
        for index in even_ones_indices:             # find max magnitude and index for even number of ones
            if (magnitude_array[index] > second_mag2):
                second_mag2 = magnitude_array[index]
                second_index2 = index
    else:
        even_ones_indices.remove(max_index1)
        
        max_index2 = -1
        max_mag2 = -1
        for index in even_ones_indices:             # find max magnitude and index for even number of ones
            if (magnitude_array[index] > max_mag2):
                max_mag2 = magnitude_array[index]
                max_index2 = index
        
        odd_ones_indices.remove(second_index1)
        
        second_index2 = -1
        second_mag2 = -1
        for index in odd_ones_indices:             # find max magnitude and index for even number of ones
            if (magnitude_array[index] > second_mag2):
                second_mag2 = magnitude_array[index]
                second_index2 = index
    
    return (max_index1, max_mag1, max_index2, max_mag2, second_index1, second_mag1, second_index2, second_index2, parity)

def get_phase_projections_1(x,deviation): #This function takes a vector of length 2**m for some positive integer m and uses intensity and relative phase measurements to return that vector up to a global phase factor
    if (len(x) == 1):                    # if vector is of length one, just return an intensity measurement of the entry
        return np.array([intensity_measurement_bipartite(x[0],deviation)])

    (m, x, power_of_two, old_length) = check_if_power_of_two(x) # extends the vector x if necessary to have lenght a power of 2
    
    magnitude_array = magnitudes(x, deviation, old_length) # take N measurements to get the magnitudes of each entry
    (max_index1, max_mag1, max_index2, max_mag2, second_index1, second_mag1, second_index2, second_mag2, parity) = largest_two_both_sides(magnitude_array)    

    x_projection = np.zeros(len(x), complex)
    
    for i in range(len(x)):
        x_projection[i]       = x[i]
    
    rel_phase_max1_sec2 = relative_phase_noise( (x[max_index1], x[second_index2]), 1, 1 ,deviation)[0]
    rel_phase_sec1_max2 = relative_phase_noise( (x[second_index1], x[max_index2]), 1, 1 ,deviation)[0]
    
    rel_phase_max1_sec2 = rel_phase_max1_sec2/np.absolute(rel_phase_max1_sec2)
    rel_phase_sec1_max2 = rel_phase_sec1_max2/np.absolute(rel_phase_sec1_max2)
#     print(rel_phase_max1_sec2/np.absolute(rel_phase_max1_sec2))
#     print(rel_phase_sec1_max2/np.absolute(rel_phase_sec1_max2))
#     print(x[max_index1], rel_phase_max1_sec2/np.absolute(rel_phase_max1_sec2)* x[second_index2])
#     print(x[second_index1], rel_phase_sec1_max2/np.absolute(rel_phase_sec1_max2)* x[max_index2])

    #altering first 2 entries of the signal with 2 projections
    x_projection[max_index1] = np.multiply((1.0/np.sqrt(2)), x[max_index1] + np.multiply(rel_phase_max1_sec2, x[second_index2]))
    x_projection[second_index2] = np.multiply((1.0/np.sqrt(2)), x[max_index1] - np.multiply(rel_phase_max1_sec2, x[second_index2]))
    max_mag1 = np.absolute(x_projection[max_index1])
    
    #altering second 2 entries of the signal with 2 projections
    x_projection[second_index1] = np.multiply((1.0/np.sqrt(2)), x[second_index1] + np.multiply(rel_phase_sec1_max2, x[max_index2]))
    x_projection[max_index2] = np.multiply((1.0/np.sqrt(2)), x[second_index1] - np.multiply(rel_phase_sec1_max2, x[max_index2]))
    second_mag1 = np.absolute(x_projection[second_index1])
    
#     print("proj", x_projection)
    
    # NEED TO CHANGE THOSE np.absolute() TO INTENSITY MEASUREMENTS
    x_temp, global_phase_factor = estimate_x(max_index1, max_mag1, second_index1, second_mag1, parity, x_projection, m, deviation, power_of_two, old_length)
    # NOT SURE WHAT TO DO ABOUT GLOBAL PHASE FACTOR WITH THE PROJECTIONS
    x_hat = np.multiply(global_phase_factor,x_temp)
#     print(np.absolute(global_phase_factor))
#     print("esti", x_hat)
    
    x_final = np.zeros(old_length, complex)
    for i in range(old_length):
        x_final[i] = x_hat[i]
    
    x_final[max_index1] = np.multiply((1.0/np.sqrt(2)), x_hat[max_index1] + x_hat[second_index2])
    x_final[second_index2] = np.multiply(np.multiply(rel_phase_max1_sec2.conjugate(), 1.0/np.sqrt(2)), x_hat[max_index1]-x_hat[second_index2])
    
    x_final[second_index1] = np.multiply((1.0/np.sqrt(2)), x_hat[second_index1] + x_hat[max_index2])
    x_final[max_index2] = np.multiply(np.multiply(rel_phase_sec1_max2.conjugate(), 1.0/np.sqrt(2)), x_hat[second_index1]-x_hat[max_index2])
    
#     print("fina", x_final)
    return x_final, 1

def test_bipartite_v_projections(file_name): # given a certain number of trials per value of N, this function evaluates the average MSE (relative error not squared relative error) of getPhaseNoiseBipartite in a way that is comparable to the paper Phase Retrieval with Polarization by Mixon
    signal_vectors, noise_vectors = get_vectors_and_noise_from_Mixon(file_name)
    
    average_RE_bipartite = []               # records the average MSE for each value of N
    average_RE_projections = []               # records the average MSE for each value of N

    for i in tqdm(range(len(signal_vectors))):
#         print(i)
        x = signal_vectors[i]   # create the signal of length N with standard deviation 1/sqrt(N)
        
        N = len(x)
        m = int(math.log(N,2))     # get m = log(N) always rounding down
        
        noise_deviation  = 0.4/np.sqrt(N)          # nu   ~ N(0, (0.4)^2 /N  )
        
        ### bipartite
        x_hat_array = []
        for j in range(8):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
            (x_hat_bipartite, gpf) = get_phase_bipartite(x, noise_deviation) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
            x_hat_array.append(x_hat_bipartite*gpf)
        
        x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
        average_RE_bipartite.append(relative_error(x_hat, x))
        
        ### projections
        x_hat_array = []
        for j in range(8):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
            (x_hat_projections, gpf) = get_phase_projections_1(x, noise_deviation) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
            x_hat_array.append(x_hat_projections*gpf)
        
        x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
        average_RE_projections.append(relative_error(x_hat, x))
        
        # end for loop
    
    x_axis = []
    for signal in signal_vectors: 
        x_axis.append(len(signal))
        
    plt.plot(x_axis, average_RE_bipartite, color = 'b', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    plt.plot(x_axis, average_RE_projections, color = 'g', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    #CHANGE
    title = "RE of Bipartite vs. Projections"
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("Relative Error")
#     upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
    plt.axis([0, 130, 0, 0.02])            # set axes
    plt.show
#     plt.savefig('graphs/Ben_bipartite_vs_projections_8.png')
    return average_RE_bipartite, average_RE_projections