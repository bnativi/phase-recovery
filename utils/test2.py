import numpy as np
from tqdm import tqdm_notebook as tqdm

from utils.random import *
from utils.relative_phase import *
from utils.relative_phase_noise import *

from utils.get_phase_original_noise import *
from utils.get_phase_ASMT import *
from utils.get_phase_bipartite import *

def relative_error(x_hat, x):
    return np.linalg.norm(x_hat - x)/np.linalg.norm(x)

def test_alg_for_variance(trials,size,magnitude,deviation): # measures the mean squared error in each entry of xhat when using getPhaseNoiseOpt   
    listy = []
    for i in tqdm(range(trials)):
        x = generate_x_uniform(size,magnitude)
        v = get_phase_noise(x,deviation)
        
        for e in range(len(x)):
            listy.append(np.linalg.norm(v-x)/np.linalg.norm(x))       # use the difference in each trial
#         print(x)
#         print(v)
#         print(computeVariance(v,x))
#         print
    return np.average(listy)

def test_rel_phase_for_variance(trials,m,magnitude,deviation):
    a = 1
    
    dic = {}
    for e in range(2**m):
        dic[e] = []
        
    for i in tqdm(range(trials)):
        x = generate_x_uniform(2**m,magnitude) #[ 0, 0, 0, 0]
        rPhase  = relative_phase(x,a,m)
        rPhaseN = relative_phase_noise(x,a,m,deviation)
#         for e in range(len(rPhase)):
#             print(rPhase[e], rPhaseN[e]
#         print
#         print(rPhaseN[0] - rPhaseN[2]
        
        for e in range(2**m):
            dic[e].append(rPhase[e]-rPhaseN[e])

    listy = []
#     print(dic
    for e in range(2**m):
        listy.append(np.var(dic[e]))
    return listy

#test code
# a = 1
# trials = 10
# magnitude = 10
# deviation = 1

# print(testRelPhasForVariance(trials,2,magnitude,deviation)

def test_opt_alg_correctness(trials,size,magnitude,deviation):  # this method was designed to check to see if a getPhase algorithm is correctly computing x
    dic = {}                                                 # this method was specifically designed to test getPhaseNoiseNew, but could be used on any iteration of getPhase
    for e in range(size):
        dic[e] = []
    for i in tqdm(range(trials)):
        x = generate-x_uniform(size,magnitude)     # x
        v = get_phase_ASMT(x,deviation) # xhat
        
        for e in range(len(x)):
            dic[e].append(v[e]-x[e])      # we only care about the differences between actual and our measured
#         print(x
#         print(v
#         print(computeVariance(v,x)
#         print
    listy = []
#     print(dic
    for e in range(size):
        listy.append(np.mean(dic[e]))     # finds the average error in each entry
    return listy

def test_ASMT_relative_error(trials,size,magnitude,deviation): # measures the mean squared error between the signal x and our estimate xhat   
    listy = []
    for i in tqdm(range(trials)):
        x = generate_x_uniform(size,magnitude)
        v = get_phase_ASMT(x,deviation)
        
        for e in range(len(x)):
            listy.append(np.linalg.norm(v-x)/np.linalg.norm(x))       # use the difference in each trial
#         print(x)
#         print(v)
#         print(computeVariance(v,x))
#         print
    return np.average(listy)

                                          # code for testing purposes
def relative_phase_noise_2(x,a,m,deviation): # relative phase method modified to check that there is a relationship between some of the measurements made    
    if (m != 1):
        print("ERROR 1: relativePhaseNoise2")
        return
    HN = make_HN(m)
    Emat = build_eigenvalue_matrix(a,m)
    mVec = build+eigenvectors_noise(x,a,m,deviation)
    if (round_complex(mVec[0]+mVec[1]-mVec[2]) == round_complex(mVec[3])):  # checks that one of the measurements is dependent on the other 3
        print("true")
    else:
        print("ERROR 2: relativePhaseNoise2")
        print
        print(x)
        print
        print
    
    return (1/np.sqrt(2**m)) * np.matmul(np.matmul(HN,Emat),mVec)

def testMSE_Original_v_Opt_v_Bipartite_Uniform(trials, size, magnitude, deviation_of_noise): # measures the mean squared error between the signal x and our estimate x_hat   
    list_original = []
    list_ASMT = []
    list_bipartite = []

    for i in tqdm(range(trials)):
        x = generate_x_uniform(size,magnitude)
        x_hat_original =         get_phase_noise(x,deviation_of_noise)
        x_hat_ASMT =              get_phase_ASMT(x,deviation_of_noise)
        (x_hat_bipartite, gpf) = get_phase_bipartite(x,deviation_of_noise)
        
        for e in range(len(x)):
            list_original.append(  np.linalg.norm(x_hat_original  - x)/np.linalg.norm(x))       # use the difference in each trial
            list_ASMT.append(       np.linalg.norm(x_hat_ASMT- x)/np.linalg.norm(x))       # use the difference in each trial
            list_bipartite.append( np.linalg.norm(x_hat_bipartite*gpf - x)/np.linalg.norm(x))       # use the difference in each trial

    return [np.average(list_original), np.average(list_ASMT), np.average(list_bipartite)]

def testMSE_Original_v_Opt_v_Bipartite_Normal(trials, size, deviation_of_x, deviation_of_noise): # measures the mean squared error between the signal x and our estimate x_hat   
    list_original = []
    list_ASMT = []
    list_bipartite = []

    for i in tqdm(range(trials)):
        x = generate_x_normal(size,deviation_of_x)
        
        x_hat_original =         get_phase_noise(x,deviation_of_noise)
        x_hat_ASMT =              get_phase_ASMT(x,deviation_of_noise)
        (x_hat_bipartite, gpf) = get_phase_bipartite(x,deviation_of_noise)
        
        for e in range(len(x)):
            list_original.append(  np.linalg.norm(x_hat_original  - x)/np.linalg.norm(x))       # use the difference in each trial
            list_ASMT.append(       np.linalg.norm(x_hat_ASMT       - x)/np.linalg.norm(x))       # use the difference in each trial
            list_bipartite.append( np.linalg.norm(x_hat_bipartite*gpf - x)/np.linalg.norm(x))       # use the difference in each trial

    return [np.average(list_original), np.average(list_ASMT), np.average(list_bipartite)]

