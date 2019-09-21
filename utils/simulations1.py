from utils.test2 import *
from tqdm import tqdm_notebook as tqdm

# Tell Python to include plots as embedded graphics.
# %matplotlib inline

# Import plotting, numpy, and library commands
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)


def NSRvsMSE(trials, startMag, startDev, increments, m, increaseFactor): # compares the effect of increasing the noise to signal ratio on the mean squared error with the original and altersignal measurement techniques         
                                                                         # we are using a fixed choice of m
    magnitude = startMag*1.0  # approximate magnitude of each x_i
    deviation = startDev*1.0  # standard deviation
    
    x = []
    y = []
    magArray = []
    
    for i in tqdm(range(increments)): 
        magArray.append((deviation**2)/(magnitude**2))
#         print(magArray
        x.append(test_alg_for_variance(int(trials), 2**m, magnitude, deviation))
        y.append(test_ASMT_relative_error(int(trials), 2**m, magnitude, deviation))
        
        magnitude = magnitude*increaseFactor
        deviation = deviation*np.sqrt(increaseFactor)
    
    print(magArray)
#     x = np.log(x)
#     y = np.log(y)
    plt.plot(magArray, x, 'b')
    plt.plot(magArray, y, 'g')
    title = "MSE with " + str(trials) + " trials and m = " + str(m)
    plt.title(title)
    plt.xlabel("Noise to Signal Ratio")
    plt.ylabel("Mean Squared Error")
    plt.show
    
# plt.axis([0, 6, 0, 20])
# color='green', marker='o', linestyle='-', linewidth=2, markersize=12

def MvsMSE(trials, trialFactor, magnitude, deviation):  # compares the effect of increasing m on the mean squared error with the original and altersignal measurement techniques         
    startm = 2
    increments = 5
    m = startm
    magnitude=magnitude*1.0
    deviation=deviation*1.0
    
    x = []
    y = []
    mArray = []
    
    for i in tqdm(range(increments)): 
        mArray.append(m)
#         print(mArray)
        x.append(test_alg_for_variance(int(trials), 2**m, magnitude, deviation))
        y.append(test_ASMT_relative_error(int(trials), 2**m, magnitude, deviation))
        
        m = m+1
        trials = trials*trialFactor  # reduce the number of trials with increasing m as increasing m lengthens the time of computation
    
    print(mArray)
#     x = np.log(x)
#     y = np.log(y)
    plt.plot(mArray, x, color = 'b', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    plt.plot(mArray, y, color = 'g', marker = 'o', linestyle = '-', linewidth =1, markersize = 6)
    nsr = (deviation**2)/(magnitude**2)
    title = "MSE with NSR = " + str(nsr)
    plt.title(title)
    plt.xlabel("m")
    plt.ylabel("Mean Squared Error")
    upperY = 7*nsr                         # this choise of upperY makes the graphs look nice
    plt.axis([1, 7, 0, upperY])            # set axes
    plt.show

# (trials, trialFactor, magnitude, deviation, increments, startm)
# MvsMSE(300, 0.333, np.sqrt(10), 1, 5, 2)