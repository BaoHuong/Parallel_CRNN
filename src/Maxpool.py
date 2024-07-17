import numpy as np
from size import Shape  


class Maxpool():
    def __init__(self, numKernel ,sizeKernel: Shape):
        self.sizeKernel = sizeKernel
        self.numKernel = numKernel
                
    def region(self, shape):
        for i in self.sizeKernel.h:
            for j in  self.sizeKernel.w:
                self.position.append(i)
        

    def forward(self, input, shape:Shape):
        resultShape = Shape(shape.h // self.sizeKernel.h, shape.w // self.sizeKernel.w)
        output = np.zeros(resultShape.h* resultShape.w)
        # print(resultShape.h, resultShape.w)
       
        for ilayer in range(self.numKernel):
            temp = input[ilayer].reshape((shape.h, shape.w))
            for ir in range(resultShape.h):
                for ic in range(resultShape.w):
                    output[ir * resultShape.w + ic] = np.max(temp[ir*self.sizeKernel.w:(ir+1)*self.sizeKernel.w,ic*self.sizeKernel.h:(ic+1)*self.sizeKernel.h])
        # print(output.reshape((resultShape.h, resultShape.w)))
        return output , resultShape



