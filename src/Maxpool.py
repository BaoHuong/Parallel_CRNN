import numpy as np
from size import Shape  
from activation import relu, softmax
from numba import cuda, jit 
import numba as nb
import time
import math


class Maxpool():
    def __init__(self, numKernel ,sizeKernel: Shape):
        self.sizeKernel = sizeKernel
        self.numKernel = numKernel
        
        
    def regions(self, a, resultShape):
        
        for i in range(resultShape.h):
            for j in range(resultShape.w):
                # print(f'a[{i* self.sizeKernel.h} : {(i + 1)* self.sizeKernel.h}, {j* self.sizeKernel.w }: {(j + 1)* self.sizeKernel.w}]')
                im_region = a[i* self.sizeKernel.h : (i + 1)* self.sizeKernel.h, 
                              j* self.sizeKernel.w : (j + 1)* self.sizeKernel.w]
                yield im_region, i, j
        
    def forward_sequential(self, input, shape:Shape):
        resultShape = Shape(shape.h // self.sizeKernel.h, shape.w // self.sizeKernel.w)
        output = np.zeros((self.numKernel, resultShape.h* resultShape.w),dtype=float)
        # print(resultShape.h, resultShape.w)
       
        for ilayer in range(self.numKernel):
            temp = input[ilayer]
            temp = temp.reshape(shape.h, shape.w)
            # for ir in range(resultShape.h):
            #     for ic in range(resultShape.w):
            #         output[ilayer][ir * resultShape.w + ic] = np.max(temp[ir*self.sizeKernel.w:(ir+1)*self.sizeKernel.w, ic*self.sizeKernel.h:(ic+1)*self.sizeKernel.h])
            for im_region, i, j in self.regions(temp,resultShape):
                output[ilayer][i * resultShape.w + j] = np.amax(im_region, axis=(0, 1))
 
        
        # print(output.reshape((resultShape.h, resultShape.w)))
        return output , (resultShape.h, resultShape.w)
    
  
def forward_sequence_pure(input, in_shape, output, out_shape, size_kernel, num_layer):
    # print(resultShape-0, resultShape.w)
    for ilayer in range(num_layer):
        temp = input[ilayer]
        temp = temp.reshape(in_shape[0], in_shape[1])
        for r in range(out_shape[0]):
            for c in range(out_shape[1]):
                max = input[ilayer][(r * size_kernel[0])*in_shape[1]  +  c * size_kernel[1]]     
                for ik in range(r * size_kernel[0], (r + 1) * size_kernel[0]): 
                    for jk in range(c * size_kernel[1] , (c + 1) * size_kernel[1]):
                        if max < input[ilayer][ik * in_shape[1] + jk]:
                            max = input[ilayer][ik * in_shape[1] + jk]
                output[ilayer][r * out_shape[1]+ c] = max

    
    return output       
    
@jit()    
def forward_sequential(input, in_shape, output, out_shape, size_kernel, num_layer):
    # print(resultShape-0, resultShape.w)
    for ilayer in range(num_layer):
        temp = input[ilayer]
        temp = temp.reshape(in_shape[0], in_shape[1])
        for r in range(out_shape[0]):
            for c in range(out_shape[1]):
                max = input[ilayer][(r * size_kernel[0])*in_shape[1]  +  c * size_kernel[1]]     
                for ik in range(r * size_kernel[0], (r + 1) * size_kernel[0]): 
                    for jk in range(c * size_kernel[1] , (c + 1) * size_kernel[1]):
                        if max < input[ilayer][ik * in_shape[1] + jk]:
                            max = input[ilayer][ik * in_shape[1] + jk]
                output[ilayer][r * out_shape[1]+ c] = max

    
    return output   
    
@cuda.jit()
def forward_paralle(input, in_shape, output, out_shape, size_kernel, num_layer):
    
    r,c = cuda.grid(2)
    # r = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (c < out_shape[1]) and (r < out_shape[0]): 
        for ilayer in range(num_layer):
            max = input[ilayer][(r * size_kernel[0])*in_shape[1]  +  c * size_kernel[1]]             
            for ik in range(r * size_kernel[0], (r + 1) * size_kernel[0]): 
                for jk in range(c * size_kernel[1] , (c + 1) * size_kernel[1]):
                    if max < input[ilayer][ik * in_shape[1] + jk]:
                        max = input[ilayer][ik * in_shape[1] + jk]
                        
            output[ilayer][r*out_shape[1]+c] = max
      
@cuda.jit()
def forward_paralle(input, in_shape, output, out_shape, size_kernel, num_layer):
    
    r,c = cuda.grid(2)
    # r = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (c < out_shape[1]) and (r < out_shape[0]): 
        for ilayer in range(num_layer):
            max = input[ilayer][(r * size_kernel[0])*in_shape[1]  +  c * size_kernel[1]]             
            for ik in range(r * size_kernel[0], (r + 1) * size_kernel[0]): 
                for jk in range(c * size_kernel[1] , (c + 1) * size_kernel[1]):
                    if max < input[ilayer][ik * in_shape[1] + jk]:
                        max = input[ilayer][ik * in_shape[1] + jk]
                        
            output[ilayer][r*out_shape[1]+c] = max
            
@cuda.jit()
def forward_3D(input, in_shape, output, out_shape, size_kernel, num_layer):
    l,r,c = cuda.grid(3)
    if (c < out_shape[1]) and (r < out_shape[0]) and (l < num_layer): 
        max = input[l][(r * size_kernel[0])*in_shape[1]  +  c * size_kernel[1]]             
        for ik in range(r * size_kernel[0], (r + 1) * size_kernel[0]): 
            for jk in range(c * size_kernel[1] , (c + 1) * size_kernel[1]):
                if max < input[l][ik * in_shape[1] + jk]:
                    max = input[l][ik * in_shape[1] + jk]          
        output[l][r*out_shape[1]+c] = max        

