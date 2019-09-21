import math
from utils.binary_operations import *
from utils.heisenberg_weyl_methods import *
from utils.relative_phase import *

# original, working on 1/22/19 for a = len(vec)/2
def get_phase_original(vec):
    length = len(vec)
#     print
#     print(vec)
#     print("length", length)
    
    if (length==1):
        return vec  #return exact entry
#         vec = np.array(vec)
#         lst = []
# #         print(math.sqrt(np.trace( np.outer(vec,vec.conj()) ))
#         lst.append(math.sqrt(np.trace( np.outer(vec,vec.conj()) )))
#         return lst
    
    m = int(math.log(length,2))
    a = int(length/2)
    if (a != length/2):
    	print("ERROR get_phase_original: length of vector is odd")
    	
    aBin = int_to_bin(a,2**m)
    
    Omeg = create_Omega_a(aBin,2**m)
    NOmeg = create_NOmega_a(aBin,2**m)
#     print(Omeg, NOmeg)
        
    halfVec = []
    for b in Omeg:
        halfVec.append(vec[bin_to_int(b)])
    recurse_vec = get_phase_original(halfVec) #
#     print
#     print(halfVec)
#     print("recurse_vec", recurse_vec)
    
    rel_phase_vec = relative_phase(vec,a,m)
#     print("rel_phase_vec", rel_phase_vec)
    
    for i in range(a):
#         print(rel_phase_vec[a+i])
#         print(recurse_vec[i])
        recurse_vec.append(rel_phase_vec[a+i] * (1.0/ recurse_vec[i].conjugate() )) #
        
    return recurse_vec