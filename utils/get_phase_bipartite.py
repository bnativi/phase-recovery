import math
import numpy as np
from utils.random import *
from utils.relative_phase_noise import *
import itertools # used for combinations and permutations

def intensity_measurement_bipartite(num,deviation): # returns the intensity measurement of a given entry of x by taking in a complex number "num" and the deviation of the noise and returning the square root of the magnitude of num plus some complex noise    
        noise = complex_random_noise(deviation)                   # get complex noise for the intensity measurment
        while (np.real(num*num.conjugate() + noise)<0):         # the real part cannot be nonnegative, if so we just get new noise (although this is not rigorous)
            noise = complex_random_noise(deviation)
        return np.sqrt(np.real(num*num.conjugate() + noise))    # gets the intensity measurement, note that we only take the real part of the noise as the intensity measurement should give a nonnegative real value

def magnitudes(array, deviation, old_length):      # makes an intensity measurment for each entry of x
    magnitudes = np.zeros(len(array))  # array to store the magnitudes estimates
    for i in range(old_length):
        magnitudes[i] = intensity_measurement_bipartite(array[i],deviation) # take each measurement
    return magnitudes

def get_indices(m):           # given a value of m, returns a list of all binary numbers in F_2^m with an even number of ones and an odd number of ones
    even_ones_indices = []
    odd_ones_indices = []
    
    even_nums = []   # stores even numbers [0,2,..., (m or m-1)]
    odd_nums = []    # stores odd numbers  [1,3,..., (m or m-1)]
    i = 0
    while (i <= m):
        if ( (i%2) == 0):
            even_nums.append(i)
        else:
            odd_nums.append(i)
        i+=1
    
    num_list = []  # stores values [1,2,4,...,2^(m-1)]
    num = 1
    for i in range(m):
        num_list.append(num)
        num = num * 2
    
    for even in even_nums:                                    # creates all binary objects of length m that have an even number of 1's
        values = list(itertools.combinations(num_list, even)) # gets permutations of an even length from the set [1,2,4,...,2^(m-1)]
        for val in values:                                    # for each binary number with "even" number of ones, add up those values to get the integer corresponding to that binary number
            integer = 0
            for i in val:
                integer += i
            even_ones_indices.append(integer)
    
    for odd in odd_nums:                                      # creates all binary objects of length m that have an odd number of 1's
        values = list(itertools.combinations(num_list, odd))  # gets permutations of an even length from the set [1,2,4,...,2^(m-1)]
        for val in values:                                    # for each binary number with "odd" number of ones, add up those values to get the integer corresponding to that binary number
            integer = 0 
            for i in val:
                integer += i
            odd_ones_indices.append(integer)
    
#     print("even indices", even_ones_indices
#     print("odd indices", odd_ones_indices
    return (even_ones_indices, odd_ones_indices)

def find_max_even_and_odd(array):       # finds the maximum magnitude for both the even_ones indices and the odd_ones indices (separately)
    m = int(math.log(len(array),2))     # compute m from the length of the vector
    (even_ones_indices, odd_ones_indices) = get_indices(m) # get the integers whose binary representations have an even number of ones and an odd number of ones respectively
    
    even_ones_index = -1
    even_ones_maximum = -1
    for index in even_ones_indices:             # find max magnitude and index for even number of ones
        if (array[index] > even_ones_maximum):
            even_ones_maximum = array[index]
            even_ones_index = index
    
    odd_ones_index = -1
    odd_ones_maximum = -1
    for index in odd_ones_indices:              # find max magnitude and index for even number of ones
        if (array[index] > odd_ones_maximum):
            odd_ones_maximum = array[index]
            odd_ones_index = index
            
    return (even_ones_index, even_ones_maximum, odd_ones_index, odd_ones_maximum)

def overall_max(even_ones_max_index, even_ones_max_mag, odd_ones_max_index, odd_ones_max_mag): #gets overall max to choose which vertex to start from
    if (even_ones_max_mag >= odd_ones_max_mag):
        return (even_ones_max_index, even_ones_max_mag, odd_ones_max_index, odd_ones_max_mag, 0)
    else:
        return (odd_ones_max_index, odd_ones_max_mag, even_ones_max_index, even_ones_max_mag, 1)

#-------------------------------------------------------------------------------------------------------------------------------------------

def check_if_power_of_two(array): # subfunction of phase recovery algorithms.  This function checks whether the length of the array is a power of 2 and extends the array to have length a power of 2 if necessary
    m = math.log(len(array),2)           # compute m from the length of the vector
    if not float(m).is_integer():        # algorithm works when length of x is a power of 2, but can work with other vectors by appending on enough 0's to make the length of x a power of 2
        m = int(round(m+0.5))                 # rounds up m to the nearest integer and records the old value of m in m_old
        final = np.zeros(2**m, complex)
        for i in range(len(array)):      # copies x to a vector of length 2**m, so its length is divisible by 2
            final[i] = array[i] 
            
        power_of_two = False             # record that the signal's length is not a power of 2
        old_length = len(array)
        
    else:                                # if the array's length is a power of 2 we just return it with the necessary information
        m = int(m)
        final = array
        power_of_two = True
        old_length = len(array)
    
    return (m, final, power_of_two, old_length)

def intensities_and_max_entries(x,deviation, old_length): # subfunction of phase recovery algorithms     
    magnitude_array = magnitudes(x, deviation, old_length) # take N measurements to get the magnitudes of each entry
    (even_ones_max_index, even_ones_max_mag, odd_ones_max_index, odd_ones_max_mag) = find_max_even_and_odd(magnitude_array)
    return overall_max(even_ones_max_index, even_ones_max_mag, odd_ones_max_index, odd_ones_max_mag)

def estimate_x(max_index, max_mag, second_index, second_mag, parity, x, m, deviation, power_of_two, old_length): # subfunction of phase recovery algorithms   
    x_hat = np.zeros(len(x), complex)     # the return of this function is the estimate x_hat
    x_hat[max_index] = max_mag       # puts the estimate of the starting vertex in x_hat
    
    if parity:                       # valid_indices1 are the indices adjacent to max_index and valid_indices2 are the indices adjacent to second_index
        (valid_indices1, valid_indices2) = get_indices(m)
    else:
        (valid_indices2, valid_indices1) = get_indices(m)
    valid_indices2.remove(max_index) # we don't want to get another estimate for max_index
    
    for index in valid_indices1:     # starting from the largest magnitude vertex, get an estimate for the phase of all entries on the opposite side of the graph with 4 * N/2 measurements, i.e. N/2 relative phase measurements
        rel_phase = relative_phase_noise( (x[max_index], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat[index] = np.multiply(rel_phase, (1.0/ x_hat[max_index].conjugate()))
        
    for index in valid_indices2:     # now from the largest magnitude vertex of the opposite side of the graph, get an estimate for the phase of all entries except the starting entry on the starting side of the graph with 4 * (N/2 - 1) measurements, i.e. N/2 - 1 relative phase measurements
        rel_phase = relative_phase_noise( (x[second_index], x[index]), 1, 1 ,deviation)[1] # should be (conjugate{x_hat[max]} * x[ind])
        x_hat[index] = np.multiply(rel_phase, (1.0/ x_hat[second_index].conjugate()))
    
    global_phase_factor = (x[max_index]/x_hat[max_index])/np.sqrt((x[max_index]/x_hat[max_index])*(x[max_index]/x_hat[max_index]).conjugate())
    
    if not power_of_two:             # if the length of the signal is not a power of 2, then there are junk entries at the end of x_hat that need to be removed
        temp = x_hat
        x_hat = np.zeros(old_length, complex)
        for i in range(old_length):
            x_hat[i] = temp[i]

    return x_hat, global_phase_factor

#-------------------------------------------------------------------------------------------------------------------

def get_phase_bipartite(x,deviation): #This function takes a vector of length 2**m for some positive integer m and uses intensity and relative phase measurements to return that vector up to a global phase factor
    if (len(x) == 1):                    # if vector is of length one, just return an intensity measurement of the entry
        return np.array([intensity_measurement_bipartite(x[0],deviation)])

    (m, x, power_of_two, old_length) = check_if_power_of_two(x) # extends the vector x if necessary to have lenght a power of 2
    (max_index, max_mag, second_index, second_mag, parity) = intensities_and_max_entries(x,deviation, old_length)
    return estimate_x(max_index, max_mag, second_index, second_mag, parity, x, m, deviation, power_of_two, old_length)


# TESTING 7/30/19 to make sure that the algorithm can handle when the length of x is not a power of 2 (by appending zeros)
# working, correctly reproduces the signal when the standard deviation of the noise is 0