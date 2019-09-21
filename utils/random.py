import numpy as np
import random as rand

def complex_random_noise(deviation): #generates a random complex number using a gaussian distribution for the real and imaginary components
                                   #this random complex number has standard deviation "deviation"
                                   #NOTE: if you want a random complex number with standard deviation 1, you should input deviation 1
    noise = deviation * np.sqrt(1.0/2.0)*(np.random.normal(0.0,1.0)+1j*np.random.normal(0.0,1.0))
#     print(noise)
    return noise
    
def generate_x_uniform(size,magnitude):    #generates a random complex vector x of length "size" and each entry has on average magnitude "magnitude"
    #create mystery vector x
    x = 1j * (np.random.rand(size)-0.5)/0.382789810647 #creates complex component first with uniform random value (-1,1)
    x += (np.random.rand(size)-0.5)/0.382789810647     #adds real part to complex with uniform random value (-1,1)
                                                       #divides by 0.382789810647 to give average magnitude 1 when input parameter "magnitude" is 1
    return magnitude*x

def generate_x_normal(size,deviation):    #generates a random complex vector x of length "size" and each entry has on average magnitude "magnitude"
    #create mystery vector x
    x = 1j * np.random.normal(0, deviation, size)     #creates complex component first drawn from a normal distribution N(0,deviation)
    x +=     np.random.normal(0, deviation, size)     #adds real part to complex drawn from a normal distribution N(0,deviation)

    return x

def intensity_measurement(e,deviation): #takes in entry "e" and standard deviation "deviation" and returns an intensity measurement
        a = np.sqrt(e**2 + complex_random_noise(deviation))    #this method was created because there are two distinct square roots of a complex number
        if (np.real(e)>0 and np.real(a)>0 or np.real(e)<0 and np.real(a)<0): # if the same sign, then a ~ e
            return a
        else:                                                                # otherwise, a ~ -e
            return -a