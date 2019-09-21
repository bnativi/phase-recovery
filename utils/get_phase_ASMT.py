import math
import numpy as np
from utils.binary_operations import *
from utils.heisenberg_weyl_methods import *
from utils.relative_phase_noise import *


#This function takes a vector of length 2**m for some positive integer m and uses relative phase and choice of a to return that vector up to a global phase factor
def get_phase_ASMT(x,deviation): #this is the culmination of all previous work and is the desired alrogithm for iteratively determining phase with relative phase measurements
#     print("x", x
    length = len(x) #we start by computing the length of the vector
#     print(length
    
    if (length==1): #if the vector has one entry, this is the base case and we either return the vector or in reality take an intensity measurement of that entry to deterimine the vector up to a global phase factor
        x = np.array(x) #return an intensity measurement
        lst = []
        lst.append(intensity_measurement(x[0],deviation)) # intensity measurement with noise
#         lst.append(np.sqrt(x[0]**2 + complexRandomNoise(deviation)) ) # intensity measurement with noise
#         print(lst
        return lst
    
    m = int(math.log(length,2)) #compute m from the length of the vector
    
    #make a choice of a
#     a = length/2   #Note that these are the only two choice of a that will always have the property that a*a=1
    a = 1
    
    aBin = int_to_bin(a,2**m)
    
    Omeg = create_Omega_a(aBin,2**m)   #create Omega_a and its complement NOmega_a for later use, each of which is a set of binary elements
    NOmeg = create_NOmega_a(aBin,2**m)
        
    halfVec = []  #this vector will consist of the entries of vec that correspond to 
    dic = {}      #this dictionary will store the conversion between entries of x to entries of halfVec 
    count = 0
    for bBin in Omeg: #add each entry of x corresponding with a bBin in Omega_a and add the entry conversion to dic
        halfVec.append( x[bin_to_int(bBin)] )
        dic[bin_to_int(bBin)] = count
        count += 1
        
    recurse_vec = get_phase_ASMT(halfVec,deviation)      #recursive call on the half of x corresponding to Omega_a
#     print("r", recurse_vec
    
    # build the vector to be fed into the relative phase computation
    # our goal here is to NOT feed in x, but instead feed in x with our measurements where we have already determined an entrie's phase
    # according to my computations, this technique should reduce the noise, i.e. it is more beneficial to 
    # have x_w\bar{\hat{ x_{w+a} }} then x_w\bar{ x_{w+a} }
    tempX = []
    for i in range(len(x)):
        bBin = int_to_bin(i,2**m)
        if (bBin in Omeg):
            tempX.append(recurse_vec[dic[i]])
        else:
            tempX.append(x[i])
    
    rel_phase_vec = relative_phase_noise(tempX,a,m,deviation) #get the relative phase measurements
    
    final = [] #the final vector that should equal x, which will be built from recurse_vec and computations involving relative phase
    
    count = int(length/2)   #this counter variable is used to count which entry of the relative phase vector should be used.  
    if (count != length/2):     #we compute phase by using the relative phase measurements from the second half of rel_phase_vec
        print("ERROR get_phase_ASMT: length of vector is odd")                 
                            #because those are of the form hat{x_{w+a}}*x_w and we can then multiply by inverse{ hat{x_{w+a}} }
    for i in range(length): 
        bBin = int_to_bin(i,2**m) #binary conversion of i to be used in the XOR function
        if (bBin in Omeg):      #if this entry is in Omega_a, then it has already been computed in recurse_vec and can be copied
            final.append(recurse_vec[dic[i]])
        else:                   #else we are considering some entry of NotOmega_a, and need to use relative phase with an entry
                                #in Omega_a
            final.append(rel_phase_vec[count] * (1.0/ recurse_vec[dic[bin_to_int(xor_int(aBin, bBin))]].conjugate() )) #
            count += 1;
        
    return final