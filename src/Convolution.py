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

    def addPadding(self, data, shape):
        width, height = shape[1], shape[0]

        newWidth = shape[1] + self.padding * 2
        newHeight = shape[0] + self.padding * 2
        newdata = np.zeros((self.numKernel, newWidth* newHeight))

        for n in range(self.numKernel):    
            for i in range(height):
                for j in range(width):
                    newdata[n][(i + self.padding)* (width+ self.padding * 2) +(j + self.padding)] = data[n][i * width + j]

            for i in range(self.padding):
                for k in range (width + self.padding * 2):
                    newdata[n][i * ( width + self.padding * 2) + k] = newdata[n][self.padding * ( width + self.padding * 2) + k ]
                    newdata[n][(height+ self.padding+ i) *( width + self.padding * 2) + k] = newdata[n][(height + self.padding - 1) * ( width + self.padding * 2) + k]
            
            for i in range(height + self.padding * 2):
                for k in range (self.padding):
                    newdata[n][i * ( width + self.padding * 2) + k] = newdata[n][i * ( width + self.padding * 2) + self.padding]
                    newdata[n][i * ( width + self.padding * 2) + k + width + self.padding] = newdata[n][i * ( width + self.padding * 2) + width + self.padding - 1]

        return newdata
            
    def forward(self, input, shape, active: str, padding = True):
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
                    for irk in range(self.sizeKernel.h):
                        for ick in range(self.sizeKernel.w):
                            
                            r = ir - self.padding + irk
                            c = ic - self.padding + ick
                            value += input[ilayer][r * w + c] * self.weight[ilayer][irk * self.sizeKernel.w + ick]
                    output[ilayer][(ir - self.padding)* shape[1] + (ic - self.padding)] = value + self.bias[ilayer]
        # print(output.reshape((shape[0], shape[1])))
        return relu(output)
    
def forward_seq(input, in_shape, in_layer,output, out_shape, num_layer, size_kernel, weight, bias):  

    for ilayer in range(num_layer):
        for ir in range(out_shape[0]):
            for ic in range(out_shape[1]):
                value = 0
                for irk in range(size_kernel[0]):
                    for ick in range(size_kernel[1]):
                        for i in range(in_layer):
                            r = ir + irk
                            c = ic + ick
                            value += input[i][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1] + ick]
                    # value = 0 if value + bias[ilayer] < 0 else value + bias[layer]
                output[ilayer][ir * out_shape[1] + ic] = 0 if value + bias[ilayer] < 0 else value + bias[ilayer]
                
    return output

@jit()
def forward(input, in_shape, output, out_shape, num_layer, size_kernel, weight, bias):  

    for ilayer in range(num_layer):
        for ir in range(out_shape[0]):
            for ic in range(out_shape[1]):
                value = 0
                for irk in range(size_kernel[0]):
                    for ick in range(size_kernel[1]):
                        r = ir + irk
                        c = ic + ick
                        value += input[ilayer][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1] + ick]
                # value = 0 if value + bias[ilayer] < 0 else value + bias[layer]
                output[ilayer][ir * out_shape[1] + ic] = 0 if value + bias[ilayer] < 0 else value + bias[ilayer]
                
    return output

@cuda.jit()
def CNN_forward(input,in_layer, in_shape, output, out_shape, num_layer, size_kernel, weight, bias):
    ir, ic = cuda.grid(2)   
    if (ir < out_shape[0]) and (ic < out_shape[1]): 
        for ilayer in range(num_layer):
            value = 0
            for irk in range(size_kernel[0]):
                for ick in range(size_kernel[1]):
                    for i in range(in_layer):
                        r = ir + irk
                        c = ic + ick
                        value += input[i][r * in_shape[1] + c] * weight[ilayer][irk * size_kernel[1]+ ick]
            output[ilayer][ir * out_shape[1] + ic] = (value + bias[ilayer]) if value + bias[ilayer] > 0 else 0

w = 128
h = 32
a = np.random.randint(3, size=(1, w*h))
aSize = Shape(h,w)

sizeKernel = Shape(3,3)
layer = Convolution(1,sizeKernel)
aAddPaddinf = layer.addPadding(a, (h,w))
w_add = w + layer.padding*2
h_add = h + layer.padding*2

# print("BIAS")
# print(layer.bias)
s_start = time.time()
output = layer.forward(aAddPaddinf, (h,w), "relu")
s_end = time.time()

out_seq = np.empty((layer.numKernel, h*w), dtype=float)

seq_start = time.time()
#CNN_forward(input, in_shape, output, out_shape, num_layer active, size_kernel, weight, bias):
forward_seq(aAddPaddinf, (h_add, w_add),1, out_seq, (h, w),1 ,(3,3), layer.weight, layer.bias)
seq_end = time.time()

output_paral_jit = np.empty((layer.numKernel, h*w), dtype=float)
jit_p_start = time.time()
#CNN_forward(input, in_shape, output, out_shape, num_layer active, size_kernel, weight, bias):
forward(aAddPaddinf, (h_add, w_add), output_paral_jit, (h, w),1 ,(3,3), layer.weight, layer.bias)
jit_p_end = time.time()


output_paral = np.empty((layer.numKernel, h*w), dtype=float)
block_size = (16, 16)
# print(f'({shape_out.h // block_size[0] + 1}, {shape_out.w // block_size[1] + 1})')
# grid_h, grid_w = shape_out.h // block_size[0] + 1,shape_out.w // block_size[1] + 1
# grid_size = (grid_h, grid_w)
grid_size = (math.ceil(h / block_size[0]),
             math.ceil(w / block_size[1]))
# print(grid_size)
p_start = time.time()
#CNN_forward(input, in_shape, output, out_shape, num_layer active, size_kernel, weight, bias):
CNN_forward[grid_size, block_size](aAddPaddinf, 1,(h_add, w_add), output_paral, (h, w),1 ,(3,3), layer.weight, layer.bias)
p_end = time.time()

print("Layer Convolution")
print(f'Time seq: \t{s_end - s_start}')
print(f'Time seq pure: \t{seq_end - seq_start}')
print(f'Time jit decorate: \t{jit_p_end - jit_p_start}')
print(f'time cuda.jit decorate: \t{p_end - p_start}')
# print(output_paral)

print(f'Error:\t{np.sum(output - output_paral)/(h*w)}')

print(f'Error:\t{np.sum(output - out_seq)/(h*w)}')

