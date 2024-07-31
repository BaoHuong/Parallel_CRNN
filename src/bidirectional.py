import numpy as np 
import numpy as np
import numba as nb
from numba import cuda, jit
import time
import math

#Activation Functions

#sigmoid
def sigmoid(X):
    return 1/(1+np.exp(-X))

#tanh 
def tanh_activation(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

#softmax activation
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    soft_exp_X = exp_X/exp_X_sum
    return soft_exp_X

class LSTM():
    def __init__(self, numhidden, inputShape, outFeature):
        self.numHidden = numhidden
        self.numFeature = inputShape[1]
        self.time_step  = inputShape[0]       
        self.outFeature = outFeature
        
        self.parameter = {"wf":np.random.randn(self.numHidden,self.numHidden + self.numFeature),
                     "bf":np.random.randn(self.numHidden),
                     "wi":np.random.randn(self.numHidden,self.numHidden + self.numFeature),
                     "bi":np.random.randn(self.numHidden),
                     "wo":np.random.randn(self.numHidden,self.numHidden + self.numFeature),
                     "bo":np.random.randn(self.numHidden),
                     "wc":np.random.randn(self.numHidden,self.numHidden + self.numFeature),
                     "bc":np.random.randn(self.numHidden),
                     "wy": np.random.randn(self.outFeature , self.numHidden),
                     "by": np.random.randn(self.outFeature,1)}
        
    def cell_forward(self, x, hidden , c):
        # hidden = np.randn(1,self.numHidden)
        # c = np.randn(self.numHidden)    
        c_pre = c
        h_pre = hidden
        for i in range(len(x)):
            concat = np.concatenate([h_pre,x[i]])
            print(concat.shape)
            concat = concat.reshape(-1,1)
            print(concat.shape)
            f_t = sigmoid(np.dot(self.parameter['wf'],concat) + self.parameter['bf'])
            i_t = sigmoid(np.dot(self.parameter['wi'],concat) + self.parameter['bi'])
            c_t = np.tanh(np.dot(self.parameter['wc'],concat) + self.parameter['bc'])
            cell_t = f_t* c_pre + i_t*c_t
            o_t = sigmoid(np.dot(self.parameter['wo'],concat) + self.parameter['bo'])
            
            h_t = o_t*np.tanh(cell_t)
            
            # c_pre = cell_t
            # h_pre = h_t
            print(f'by:{self.parameter["by"].shape}')
            print(f'wy:{self.parameter["wy"].shape}')
            print(f'h_t:{h_t.shape}')
            
            y_t =softmax(np.dot(self.parameter["wy"], h_t) + self.parameter["by"])
        return f_t, i_t,c_t, o_t,h_t, cell_t, y_t