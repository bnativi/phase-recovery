import numpy as np
from utils.binary_operations import *
from utils.heisenberg_weyl_methods import *
from utils.relative_phase import *
from utils.random import *

def build_eigenvectors_noise(x,a,m,deviation):  #builds the vector of intensity measurements with eigenvectors
    final = []
    x = np.array(x)
    
    (basisSa,basisTa,gSa,gTa) = get_Sa_Ta_bases(a,m)
    
    for h in np.transpose(basisSa):
        mat = np.matmul( np.outer(x,x.conj()) , np.outer(h,h.conj())) #xx^thh^t   w/ conj   x,h
        final.append( np.trace(mat) + complex_random_noise(deviation) )                             #|<x,h>|^2 = Tr(xx^thh^t)
        
    for f in np.transpose(basisTa):
        mat = np.matmul( np.outer(x,x.conj()) , np.outer(f,f.conj())) #xx^tff^t   w/ conj   x,f
        final.append( np.trace(mat) + complex_random_noise(deviation) )                                   #|<x,f>|^2 = Tr(xx^tff^t)
    
    return np.transpose(np.array(final))

def relative_phase_noise(x,a,m,deviation): #returns the relative phases of a vector x for a given choice of a
    HN = make_HN(m)
    Emat = build_eigenvalue_matrix(a,m)
    mVec = build_eigenvectors_noise(x,a,m,deviation)
    
    return (1/np.sqrt(2**m)) * np.matmul(np.matmul(HN,Emat),mVec)