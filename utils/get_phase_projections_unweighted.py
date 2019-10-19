import math
import numpy as np
import itertools # used for combinations and permutations
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

from utils.random import *
from utils.relative_phase_noise import *
from utils.get_phase_bipartite import *
from utils.Mixon_comparison_2 import *
from utils.test2 import *

def order_magnitudes(array):    # input is the array of magnitudes
                                # output is a dictionary where the keys are the indices k of the mangitudes ordered from high to low and the values are the index of array with the (k+1)th largest magnitude
    (even_ones_indices, odd_ones_indices) = get_indices(int(np.log2(len(array))))
    
    even_list = []
    for i in even_ones_indices:
        even_list.append( [i,array[i]] )

    even_list_ordered = [(even_list[elem[1]][0]) for elem in np.argsort(even_list, axis=0)] # sort
    even_magnitude_indexing = {}
    for i in range(len(even_list_ordered)):
        even_magnitude_indexing[len(even_list_ordered)-i-1] = even_list_ordered[i] # dictionary where keys are ordering low to high and values are the index of array
        
    odd_list = []
    for i in odd_ones_indices:
        odd_list.append( [i,array[i]] )
    
    odd_list_ordered = [(odd_list[elem[1]][0]) for elem in np.argsort(odd_list, axis=0)] # sort
    odd_magnitude_indexing = {}
    for i in range(len(odd_list_ordered)):
        odd_magnitude_indexing[len(odd_list_ordered)-i-1] = odd_list_ordered[i] # dictionary where keys are ordering low to high and values are the index of array
    
    return even_magnitude_indexing, odd_magnitude_indexing

#TESTING 10/17/19
# print(order_magnitudes([20, 60, 50, 40, 30, 80, 10, 70]))

def projection_unweighted(array, even_magnitude_indexing, odd_magnitude_indexing, deviation, entries_to_combine): # this function outputs a projection of the signal with the goal of increasing the magnitude of the largest entries on each side of the bipartite graph
    k = entries_to_combine # k is the number of entries we use from the opposite side of the graph to combine with the largest entry
    
    rel_phase_1 = np.zeros(k+1, complex) # vector for relative phase relating largest even to the odds
    rel_phase_2 = np.zeros(k+1, complex) # vector for relative phase relating largest odd to the evens
    
    if (k == 0): # if we are not using any additional entries and thus no projections
        return array, rel_phase_1, rel_phase_2
    
    x_projection = np.zeros(len(array), complex) # this vector will be returned
    for i in range(len(array)): # initialize to the signal for the entries that will be unchanged by the projections
        x_projection[i] = array[i]
    
    for i in range(k): # get relative phases
        # work with i+1 because i=0 is largest entry
        rel_phase_temp_1 = relative_phase_noise( (array[even_magnitude_indexing[0]], array[odd_magnitude_indexing[i+1]]), 1, 1 ,deviation)[0]
        rel_phase_temp_2 = relative_phase_noise( (array[odd_magnitude_indexing[0]],  array[even_magnitude_indexing[i+1]]), 1, 1 ,deviation)[0]
    
        rel_phase_1[i+1] = rel_phase_temp_1/np.absolute(rel_phase_temp_1)
        rel_phase_2[i+1] = rel_phase_temp_2/np.absolute(rel_phase_temp_2)
    
#     print(rel_phase_1)
#     print(rel_phase_2)

    # projections using largest even
    temp = array[even_magnitude_indexing[0]]
    for i in range(k):
        temp = temp + rel_phase_1[i+1]*array[odd_magnitude_indexing[i+1]]
    x_projection[even_magnitude_indexing[0]] = temp/np.sqrt(k+1) # get projected value of the constructive combination, hopefully is larger than before
    
    for i in range(k): # get the projected values for each of the destructive combinations
        temp = array[even_magnitude_indexing[0]]
        for j in range(k):
            if (j==i):
                temp = temp - rel_phase_1[j+1]*array[odd_magnitude_indexing[j+1]]
            else:
                temp = temp + rel_phase_1[j+1]*array[odd_magnitude_indexing[j+1]]
        
        x_projection[odd_magnitude_indexing[i+1]] = temp/np.sqrt(k+1)
    
    # projections using largest odd
    temp = array[odd_magnitude_indexing[0]]
    for i in range(k): 
        temp = temp + rel_phase_2[i+1]*array[even_magnitude_indexing[i+1]]
    x_projection[odd_magnitude_indexing[0]] = temp/np.sqrt(k+1) # get projected value of the constructive combination
    
    for i in range(k): # get the projected values for each of the destructive combinations
        temp = array[odd_magnitude_indexing[0]]
        for j in range(k):
            if (j==i):
                temp = temp - rel_phase_2[j+1]*array[even_magnitude_indexing[j+1]]
            else:
                temp = temp + rel_phase_2[j+1]*array[even_magnitude_indexing[j+1]]
        
        x_projection[even_magnitude_indexing[i+1]] = temp/np.sqrt(k+1)
    
#     print("projection", x_projection)
    
    return x_projection, rel_phase_1, rel_phase_2

def get_phase_projections_unweighted(x, deviation, entries_to_combine): #This function takes a vector of length 2**m for some positive integer m and uses intensity and relative phase measurements to return that vector up to a global phase factor
    if (len(x) == 1):                    # if vector is of length one, just return an intensity measurement of the entry
        return np.array([intensity_measurement_bipartite(x[0],deviation)])

    (m, x, power_of_two, old_length) = check_if_power_of_two(x) # extends the vector x if necessary to have lenght a power of 2
    
    magnitude_array = magnitudes(x, deviation, old_length) # take N measurements to get the magnitudes of each entry
    (even_magnitude_indexing, odd_magnitude_indexing) = order_magnitudes(magnitude_array) # returns two dictionaries that give the ordering of the magnitudes from largest to smallest.  Keys are indices 0 to N/2 and values are the index corresponds to the (i+1)th largest magnitude 
    
    x_projection, rel_phase_1, rel_phase_2 = projection_unweighted(x, even_magnitude_indexing, odd_magnitude_indexing, deviation, entries_to_combine)
    
    x_temp, global_phase_factor = estimate_x(even_magnitude_indexing[0], 
                                             intensity_measurement(x_projection[even_magnitude_indexing[0]],deviation), # intensity measurement for one side's largest entry 
                                             odd_magnitude_indexing[0], 
                                             intensity_measurement(x_projection[odd_magnitude_indexing[0]],deviation), # intensity measurement for the other side's largest entry
                                             0,  # CHANGE TO START FROM WHICHEVER SIDE IS LARGER
                                             x_projection, 
                                             m, 
                                             deviation, 
                                             power_of_two, 
                                             old_length)
    
    x_hat = np.multiply(global_phase_factor,x_temp) # CAN I STILL RECOVER ORIGINAL SIGNAL IF I CAN'T CORRECT BY GLOBAL PHASE FACTOR?
#     print(np.absolute(global_phase_factor))
#     print("esti", x_hat)
    
    x_final = np.zeros(old_length, complex) # the returned estimate of the signal
    for i in range(old_length):
        x_final[i] = x_hat[i]
    
    k = entries_to_combine
    if (k == 0): # if no projections used, the x_hat is our estimate
        return x_final, 1
    
    temp = (2-k)*x_hat[even_magnitude_indexing[0]] # estimate the largest even 
    for i in range(k):
        temp = temp + x_hat[odd_magnitude_indexing[i+1]]
    x_final[even_magnitude_indexing[0]] = (np.sqrt(k+1)/2.0)*temp
#     print(x[even_magnitude_indexing[0]], x_final[even_magnitude_indexing[0]])
    
    temp = (2-k)*x_hat[odd_magnitude_indexing[0]] # estimate the largest odd
    for i in range(k):
        temp = temp + x_hat[even_magnitude_indexing[i+1]]
    x_final[odd_magnitude_indexing[0]] = (np.sqrt(k+1)/2.0)*temp
#     print(x[odd_magnitude_indexing[0]], x_final[odd_magnitude_indexing[0]])

    for i in range(k): # estimate all other entries that were used for the projections
        x_final[odd_magnitude_indexing[i+1]]  = (np.sqrt(k+1)/(2.0*rel_phase_1[i+1]))*(x_hat[even_magnitude_indexing[0]] - x_hat[odd_magnitude_indexing[i+1]])
        x_final[even_magnitude_indexing[i+1]] = (np.sqrt(k+1)/(2.0*rel_phase_2[i+1]))*(x_hat[odd_magnitude_indexing[0]] - x_hat[even_magnitude_indexing[i+1]])

    return x_final, 1

def test_projections_unweighted_varying_k(file_name, repeats): # given a certain number of trials per value of N, this function evaluates the average MSE (relative error not squared relative error) of getPhaseNoiseBipartite in a way that is comparable to the paper Phase Retrieval with Polarization by Mixon
	signal_vectors, noise_vectors = get_vectors_and_noise_from_Mixon(file_name)
    
	average_RE_projections_0 = []               # records the average MSE for each value of N (note this is just the bipartite algorithm if no projections are used)
	average_RE_projections_1 = []               # records the average MSE for each value of N
	average_RE_projections_2 = []               # records the average MSE for each value of N
	average_RE_projections_3 = []               # records the average MSE for each value of N
	average_RE_projections_8 = []               # records the average MSE for each value of N

	for i in tqdm(range(len(signal_vectors))):
#         print(i)
		x = signal_vectors[i]   # create the signal of length N with standard deviation 1/sqrt(N)
        
		N = len(x)
		m = int(math.log(N,2))     # get m = log(N) always rounding down
        
		noise_deviation  = 0.4/np.sqrt(N)          # nu   ~ N(0, (0.4)^2 /N  )
        
        ### bipartite k=0
		x_hat_array = []
		for j in range(repeats):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
			(x_hat_bipartite, gpf) = get_phase_projections_unweighted(x, noise_deviation, 0) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
			x_hat_array.append(x_hat_bipartite*gpf)
        
		x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
		average_RE_projections_0.append(relative_error(x_hat, x))
        
        ### projections k=1
		x_hat_array = []
		for j in range(repeats):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
			(x_hat_projections1, gpf) = get_phase_projections_unweighted(x, noise_deviation, 1) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
			x_hat_array.append(x_hat_projections1*gpf)
        
		x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
		average_RE_projections_1.append(relative_error(x_hat, x))

        ### projections k=2
		x_hat_array = []
		for j in range(repeats):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
			(x_hat_projections2, gpf) = get_phase_projections_unweighted(x, noise_deviation, 2) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
			x_hat_array.append(x_hat_projections2*gpf)
        
		x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
		average_RE_projections_2.append(relative_error(x_hat, x))
        
        ### projections k=3
		x_hat_array = []
		for j in range(repeats):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
			(x_hat_projections3, gpf) = get_phase_projections_unweighted(x, noise_deviation, 3) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
			x_hat_array.append(x_hat_projections3*gpf)
        
		x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
		average_RE_projections_3.append(relative_error(x_hat, x))

        ### projections k=8 for N=20 to 128
		if(i>2):
			x_hat_array = []
			for j in range(repeats):                      # make log(N) estimates of the signal for a total of O(NlogN) measurements like in Mixon paper
				(x_hat_projections3, gpf) = get_phase_projections_unweighted(x, noise_deviation, 8) # run the algorithm with noise standard deviation of sigma/sqrt(N) where sigma = 0.4
				x_hat_array.append(x_hat_projections3*gpf)
	        
			x_hat = np.average(x_hat_array, axis=0) # average the log(N) estimates for each entry             
			average_RE_projections_8.append(relative_error(x_hat, x))
		else:
			average_RE_projections_8.append(0)
        
        # end for loop
    
	x_axis = []
	for signal in signal_vectors: 
		x_axis.append(len(signal))
        
	plt.plot(x_axis, average_RE_projections_0, color = 'royalblue', marker = 'o', linestyle = '-', linewidth =1, markersize = 3)
	plt.plot(x_axis, average_RE_projections_1, color = 'cadetblue', marker = 'o', linestyle = '-', linewidth =1, markersize = 3)
	plt.plot(x_axis, average_RE_projections_2, color = 'mediumseagreen', marker = 'o', linestyle = '-', linewidth =1, markersize = 3)
	plt.plot(x_axis, average_RE_projections_3, color = 'yellowgreen', marker = 'o', linestyle = '-', linewidth =1, markersize = 3)
	plt.plot(x_axis, average_RE_projections_8, color = 'gold', marker = 'o', linestyle = '-', linewidth =1, markersize = 3)

    #CHANGE
	title = "RE of Projections for k=0,1,2,3"
	plt.title(title)
	plt.xlabel("N")
	plt.ylabel("Relative Error")
#     upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
	plt.axis([0, 130, 0, 0.02])            # set axes
	plt.show
    plt.savefig('graphs/Ben_projections_unweighted_128_k=0,1,2,3,8.pdf')
	return average_RE_projections_0, average_RE_projections_1, average_RE_projections_2, average_RE_projections_3