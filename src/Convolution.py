import numpy as np
from size import Shape  
from activation import relu, softmax
from numba import cuda, jit
import math
import time

class Convolution():
    def __init__(self, numKernel ,sizeKernel: Shape):
        self.numKernel = numKernel
        self.sizeKernel = sizeKernel
        self.weight = np.random.randint(-1, 2, size=(numKernel, sizeKernel.w * sizeKernel.h))
        self.bias = np.random.rand(numKernel)
        self.padding = int(sizeKernel.h / 2)

    def addPadding(self, data, shape, in_layer):
        width, height = shape[1], shape[0]

        newWidth = shape[1] + self.padding * 2
        newHeight = shape[0] + self.padding * 2
        newdata = np.zeros((in_layer, newWidth * newHeight))

        for n in range(in_layer):    
            for i in range(height):
                for j in range(width):
                    newdata[n][(i + self.padding)*newWidth+(j + self.padding)] = data[n][i * width + j]

            for i in range(self.padding):
                for k in range (width + self.padding * 2):
                    newdata[n][i * ( width + self.padding * 2) + k] = newdata[n][self.padding * ( width + self.padding * 2) + k ]
                    newdata[n][(height+ self.padding+ i) *( width + self.padding * 2) + k] = newdata[n][(height + self.padding - 1) * ( width + self.padding * 2) + k]
            
            for i in range(height + self.padding * 2):
                for k in range (self.padding):
                    newdata[n][i * ( width + self.padding * 2) + k] = newdata[n][i * ( width + self.padding * 2) + self.padding]
                    newdata[n][i * ( width + self.padding * 2) + k + width + self.padding] = newdata[n][i * ( width + self.padding * 2) + width + self.padding - 1]

        return newdata
            
    def forward(self, input, shape, in_layer, active: str, padding = True):
        if active not in ['softmax', 'relu']:
            print("Error: Activation function not active")
            return
        output = np.zeros((self.numKernel, shape[1] * shape[0]), dtype=float)
        w, h = shape[1], shape[0]
        if padding:
            w = shape[1] + self.padding*2
            h = shape[0] + self.padding*2
        

        for ilayer in range(self.numKernel):
            for ir in range(self.padding, self.padding + shape[0]):
                for ic in range(self.padding, self.padding+ shape[1]):
                    value = 0
                    for i in range(in_layer):
                        for irk in range(self.sizeKernel.h):
                            for ick in range(self.sizeKernel.w):
                                r = ir - self.padding + irk
                                c = ic - self.padding + ick
                                value += input[i][r * w + c] * self.weight[ilayer][irk * self.sizeKernel.w + ick]
                    output[ilayer][(ir - self.padding)* shape[1] + (ic - self.padding)] = value + self.bias[ilayer]
        # print(output.reshape((shape[0], shape[1])))
        return relu(output)
    
def forward_seq(input, in_shape, in_layer,output, out_shape, num_layer, size_kernel, weight, bias):  
    for ilayer in range(num_layer):
        for ir in range(out_shape[0]):
            for ic in range(out_shape[1]):
                value = 0
                for i in range(in_layer):
                    for irk in range(size_kernel[0]):
                        for ick in range(size_kernel[1]):
                            r = ir + irk
                            c = ic + ick
                            value += input[i][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1] + ick]
                    # /alue = output[ilayer][ir * out_shape[1] + ic]
                # value += bias[ilayer]
                output[ilayer][ir * out_shape[1] + ic] = 0 if value + bias[ilayer] < 0 else value + bias[ilayer]
    return output

@jit()
def forward(input, in_shape, in_layer, output, out_shape, num_layer, size_kernel, weight, bias):  

    for ilayer in range(num_layer):
        for ir in range(out_shape[0]):
            for ic in range(out_shape[1]):
                value =0
                for i in range(in_layer):
                    for irk in range(size_kernel[0]):
                        for ick in range(size_kernel[1]):
                            r = ir + irk
                            c = ic + ick
                            value += input[i][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1] + ick]
                output[ilayer][ir * out_shape[1] + ic] = 0 if value + bias[ilayer] < 0 else value + bias[ilayer]
                
    return output

@cuda.jit()
def CNN_forward(input, in_shape, in_layer, output, out_shape, num_layer, size_kernel, weight, bias):
    ir, ic = cuda.grid(2)   
    if (ir < out_shape[0]) and (ic < out_shape[1]): 
        for ilayer in range(num_layer):
            value = 0
            for i in range(in_layer):
                for irk in range(size_kernel[0]):
                    for ick in range(size_kernel[1]):
                        r = ir + irk
                        c = ic + ick
                        value += input[i][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1]+ ick]
            output[ilayer][ir * out_shape[1] + ic] = (value + bias[ilayer]) if value + bias[ilayer] > 0 else 0

# if version parallel use 2-dimension-gird, this parallelism version use gird 3d to optimize
# In version 2D, the variable row and column correspond to the output index of one layer. We must use a loop to calculate the result of all layer
# Follow this way don't use GPU effectively, so instead of using 2d block, we can use 3d block. 
# In using block 3d, indexes are layer, row, and column in the matrix output result 
@cuda.jit()
def CNN_forward_3D_v1(input, in_shape, in_layer, output, out_shape, num_layer, size_kernel, weight, bias):
    il, ir, ic = cuda.grid(3)   
    if (ir < out_shape[0]) and (ic < out_shape[1]) and (il < num_layer): 
        value = 0
        for i in range(in_layer):
            for irk in range(size_kernel[0]):
                for ick in range(size_kernel[1]):
                    r = ir + irk
                    c = ic + ick
                    value += input[i][r * in_shape[1] + c] * weight[il][irk * size_kernel[1]+ ick]
        output[il][ir * out_shape[1] + ic] = (value + bias[il]) if value + bias[il] > 0 else 0


@cuda.jit()
def CNN_sharemem_v1(input, in_shape, in_layer, output, out_shape, num_layer, size_kernel, weight, bias):
    il, ir, ic = cuda.grid(3)   
    shared = cuda.shared.array((32, 32), dtype=input.dtype)
    icol = ic - size_kernel[0] + 1
    irow = ir - size_kernel[0] +1
    
    if (ir < out_shape[0]) and (ic < out_shape[1])and(il< in_layer):
        for layer in range(in_layer):
            shared[il][ir * in_shape[1] + ic] = input[il][ir * in_shape[1] + ic]        
    cuda.synchronize()
        
    if (ir < out_shape[0]) and (ic < out_shape[1]) and (il < num_layer): 
        value = 0
        for i in range(in_layer):
            for irk in range(size_kernel[0]):
                for ick in range(size_kernel[1]):
                    r = ir + irk
                    c = ic + ick
                    value += input[i][r * in_shape[1] + c] * weight[il][irk * size_kernel[1]+ ick]
        output[il][ir * out_shape[1] + ic] = (value + bias[il]) if value + bias[il] > 0 else 0


# @cuda.jit()
# def CNN_forward_3D_v2(input, in_shape, in_layer, output, out_shape, num_layer, size_kernel, weight, bias):
#     ir, ic, ivalue = cuda.grid(3)   
#     if (ir < out_shape[0]) and (ic < out_shape[1]) and (ivalue < in_layer): 
#         for ilayer in range(num_layer):
#             value = 0
#             for irk in range(size_kernel[0]):
#                 for ick in range(size_kernel[1]):
#                     r = ir + irk
#                     c = ic + ick
#                     value += input[ivalue][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1]+ ick]
#             output[ilayer][ir * out_shape[1] + ic] = (value + bias[ilayer]) if value + bias[ilayer] > 0 else 0
