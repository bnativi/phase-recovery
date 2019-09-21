import numpy as np
from utils.binary_operations import *

# E(a,b) for m tuples a,b
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

#updated by going in the reverse order of the notes as my binary strings are written low-> high and the notes writes them high -> low
def D(a_var,b_var): #takes in tuples a,b \in F_2 ^m and outputs D(a,b)
    I = np.array([[1,0],[0,1]]);    #identity I_2
    if (a_var[0] > 1 or b_var[0] > 1):      #a and b should be binary tuples
        print("ERROR 1: D(a,b): invalid a or b", a_var, b_var)
        return I
    if (a_var[0] == 1):           #g = X^a[0]
        I = np.matmul(I,X)  
    if (b_var[0] == 1):
        I = np.matmul(I,Z)    #g = X^a[0] Z^b[0]
    if (len(a_var) <=1 ):
        return I
    else:
        return np.kron(I,D(a_var[1:],b_var[1:]))  #recursively returns g kronecker with the kronecker product of all the remaining g's

def E(a_var,b_var):     #creates E(a,b) = i^{ab^T} D(a,b)
    d = D(a_var,b_var)
    val = np.matmul(a_var, np.transpose(b_var)) #computes ab^T
    val = val % 4
    #print(val, (1j)**val
    if (val == 0):
        return d
    return d * ( (1j)**val ) 

# #method for omega a
def create_Omega_a(aBin,N): #creates the set Omega_a for those binary tuples b such that ab^T = 0
    final = []
    for b in range(N):
        bBin = int_to_bin(b,N)
        
        product = 0         
        for i in range(len(aBin)):        #computes the product ab^T mod 2 (as it is binary)
            product += aBin[i] * bBin[i]
        product = product % 2
        
        if (product == 0):                #checks ab^T = 0
            final.append(bBin)
    
    return final                          #returns a list of b's 

# #method for creating the set of all b not in omega a
def create_NOmega_a(aBin,N): #creates the set NOmega_a for those binary tuples b such that ab^T = 1
    final = []
    for b in range(N):
        bBin = int_to_bin(b,N)
        
        product = 0         
        for i in range(len(aBin)):        #computes the product ab^T mod 2 (as it is binary)
            product += aBin[i] * bBin[i]
        product = product % 2
        
        if (product == 1):                #checks ab^T = 0
            final.append(bBin)
    
    return final                          #returns a list of b's 

def symp_inner_prod(a_var,b_var,m_var): #does the symplectic inner product 
    return (np.dot(a_var[0:m_var],b_var[m_var:2*m_var])+np.dot(a_var[m_var:2*m_var],b_var[0:m_var])) % 2

def make_HN(m_var): #constructs H_N = H_2 x ... x H_2 m times, where x represents the kronecker product
    H2 = np.array([[1,1],[1,-1]]) * (1.0/(2.0)**(0.5))
    if (m_var<0):
        print("ERROR 1: makeHN: m is too small", m)
        return np.array([])
    if (m_var==0):
        return np.array([1])
    if (m_var==1): #base case, it just returns H_2
        return H2 
    return np.kron(H2,make_HN(m_var-1)) #works recursively

def make_IN(m_var): #constructs I_N = I_2 x ... x I_2 m times, where x represents the kronecker product
    I2 = np.array([[1,0],[0,1]])
    if (m_var<0):
        print("ERROR 1: makeIN: m is too small", m)
        return np.array([])
    if (m_var==0):
        return np.array([1])
    if (m_var==1): #base case, it just returns H_2
        return I2 
    return np.kron(I2,make_IN(m_var-1)) #works recursively