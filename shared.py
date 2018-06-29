import sys
from types import ModuleType
from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np
import pdb
import time
import os

temp_module_name = '__hogwildsgd__temp__'

this = sys.modules[__name__]
this.sleep_time = 120
this.start_time = time.time()

class SharedWeights:
    """ Class to create a temporary module with the gradient function inside
        to allow multiprocessing to work for async weight updates.
    """
    def __init__(self, size_w):
        coef_shared = Array(c_double, 
                (np.random.normal(size=(size_w,1)) * 1./np.sqrt(size_w)).flat,
                lock=False)
        w = np.frombuffer(coef_shared)
        w = w.reshape((len(w),1)) 
        self.w = w

    def __enter__(self, *args):
        # Make temporary module to store shared weights
        mod = ModuleType(temp_module_name)
        mod.__dict__['w'] =  self.w
        sys.modules[mod.__name__] = mod    
        self.mod = mod    
        return self
    
    def __exit__(self, *args):
        # Clean up temporary module
        del sys.modules[self.mod.__name__]         


def mse_gradient_step(X, y, learning_rate):
    """ Gradient for mean squared error loss function. """
    w = sys.modules[temp_module_name].__dict__['w']
    #pdb.set_trace()
    #print('w is: ', w)

    # Calculate gradient
    err = y.reshape((len(y),1))-np.dot(X,w)
    grad = -2.*np.dot(np.transpose(X),err)/ X.shape[0]
    #pdb.set_trace()

    for index in np.where(abs(grad) > .01)[0]:#range(len(grad)):
         #print('gradient: \n', grad)
         #print('index is: ', index)
         #print('w is: \n', w)
         #print('time since beginning: ', (time.time()-this.start_time))
         #print('******************* pid: ', os.getpid(), '*******************')
         #time.sleep(get_sleep_time())
         w[index] -= learning_rate*grad[index,0]

def get_sleep_time():
    this.sleep_time -= this.sleep_time/5
    return this.sleep_time

