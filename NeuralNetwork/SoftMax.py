### SoftMax function for multiclass classification problem

# z layer output, vector in the shape of n-number of neurons in the layer
import numpy as np


def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
  
    a = np.zeros(len(z))
    
    z_sum = 0
    
    for num in range(len(z)):
        z_sum+= np.exp(z[num])
        
    for j in range(len(a)):
        a[j] = np.exp(z[j])/z_sum
    return a

test1 = np.array([1,2,3,4])
print(test1)

f = my_softmax(test1)
print(f)
np.sum(f) == 1 # should be true