import numpy as np

#creates a list of the pairs from a given choice of a and N= 2^m

def reverse_bin(binary):
    final = []
    for i in range(len(binary)):
        final.append(binary[ len(binary)-i-1 ])
    return final

def int_to_bin(num,N):  #converts an integer to a binary list of length log(N)==m
    binary = []       	#binary is read left==N==2^m to right==2^0
    if (num<0): 		#can't use negative values of num
        print("ERROR: intToBin 1; num<0", num)
        return []
    if (num>=N):       	#shouldn't input a number larger than can be represented in log(N) bits
        print("ERROR: intToBin 2; num>N;", num, N)
        return []
    while (N>1):
        if (num%2 == 1):
            binary.append(1)
            num = (num-1)/2
        else:
            binary.append(0)
            num = num/2
        N = N/2
    return reverse_bin(binary) #reversed so that leftmost bit is the largest, in accor

def bin_to_int(binary):   #converts binary to an integer
    binary = reverse_bin(binary)
    num = 0
    if (binary is None):
        print("ERROR: binToInt 1: binary is None")
        return 0
    for i in range(len(binary)):
        if (binary[i]): #note binary is a list of booleans, where True == 1
            num += 2**i
    return num

def xor_int(one, two):   #xor's two binary numbers represented by 1's and 0's
    final = []
    
    if (len(one) != len(two)):
        print("ERROR: xorBin 1: addBin lengths not equal;", len(one), len(two))
        return []
    
    for i in range(len(one)):
        if (one[i]==1 and two[i]==1): # (1 , 1) = 0
            final.append(0)
        elif(one[i]==1 or two[i]==1): # (1 , 0) = (0 , 1) = 1
            final.append(1)
        else:                   # (0 , 0) = 0
            final.append(0)
    return final
    