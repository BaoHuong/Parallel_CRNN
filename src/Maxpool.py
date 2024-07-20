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
        print(resultShape.h, resultShape.w)
       
        for ilayer in range(self.numKernel):
            temp = input[ilayer]
            temp = temp.reshape(shape.h, shape.w)
            # for ir in range(resultShape.h):
            #     for ic in range(resultShape.w):
            #         output[ilayer][ir * resultShape.w + ic] = np.max(temp[ir*self.sizeKernel.w:(ir+1)*self.sizeKernel.w, ic*self.sizeKernel.h:(ic+1)*self.sizeKernel.h])
            for im_region, i, j in self.regions(temp,resultShape):
                output[ilayer][i * resultShape.w + j] = np.amax(im_region, axis=(0, 1))
 
        
        print(output.reshape((resultShape.h, resultShape.w)))
        return output , resultShape
    
    # @cuda.jit
    # def forward_paralle(self, input, in_shape:Shape, output, out_shape):
        
    #     c, r = cuda.grid(2)

    #     if (c < out_shape.w) and (r < out_shape.h): 
    #         for ilayer in range(self.numKernel):
    #             temp = input[ilayer]
    #             temp = temp.reshape(in_shape.h, in_shape.w)                
    #             output[ilayer][r * out_shape.w + c] = np.max(temp[r * self.sizeKernel.h : (r + 1) * self.sizeKernel.h, 
    #                                                             c * self.sizeKernel.w : (c + 1) * self.sizeKernel.w])
    
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
    
@jit(cache=True)    
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
    
@cuda.jit
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
                        


layer = Maxpool(1,Shape(2,2))

h = 32
w = 128

a = np.random.randint(0, 256 ,(1, w * h)) 
# print(a.reshape(h, w))
a_size = Shape(h, w)

s_start = time.time()
output_seq, shape_out = layer.forward_sequential(a, a_size)
s_end = time.time()
# print(output_seq)
# print(f'({shape_out.h},{shape_out.w})' )
# print(f'sequential time: {s_end - s_start}')


output_paral = np.empty((layer.numKernel, shape_out.w*shape_out.h), dtype=float)
block_size = (32, 16)
# print(f'({shape_out.h // block_size[0] + 1}, {shape_out.w // block_size[1] + 1})')
# grid_h, grid_w = shape_out.h // block_size[0] + 1,shape_out.w // block_size[1] + 1
# grid_size = (grid_h, grid_w)
grid_size = (math.ceil(shape_out.h / block_size[0]),
             math.ceil(shape_out.w / block_size[1]))
# print(grid_size)

d_a = cuda.to_device(a)
start = time.time()
forward_paralle[grid_size, block_size](d_a, (h, w), output_paral, (shape_out.h, shape_out.w), (2,2), 1)
end = time.time()

output_jit = np.empty((layer.numKernel, shape_out.w*shape_out.h), dtype=float)
jit_start_1 = time.time()
forward_sequential(a, (h, w), output_jit, (shape_out.h, shape_out.w), (2,2), 1)
jit_end_1 = time.time()


jit_start = time.time()
forward_sequential(a, (h, w), output_jit, (shape_out.h, shape_out.w), (2,2), 1)
jit_end = time.time()


pure_start = time.time()
forward_sequence_pure(d_a, (h, w), output_paral, (shape_out.h, shape_out.w), (2,2), 1)
pure_end = time.time()

print('\n\nLayer Maxpool')
print(f'sequential time:\t{s_end - s_start}')
print(f'time seq pure:\t{pure_end - pure_start}')
print(f'time parallel jit decorate:\t{jit_end_1 - jit_start_1}')
print(f'time parallel jit decorate + cache:\t{jit_end - jit_start}')
print(f'time parallel cuda.jit decorate:\t{end - start}')


print(f'Error seq and cuda: \t{np.sum(abs(output_paral- output_seq))}')
print(f'Errorseq and jit: \t{np.sum(abs(jit_end - jit_end))}')
