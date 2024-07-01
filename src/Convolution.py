import numpy as np
from size import Shape  


class Convolution():
    def __init__(self, numKernel ,sizeKernel: Shape):
        self.numKernel = numKernel
        self.sizeKernel = sizeKernel
        self.weight = np.random.randint(2, size=(numKernel, sizeKernel.w * sizeKernel.h))
        self.bias = np.random.rand(numKernel)
        self.padding = int(sizeKernel.h / 2)

    def addPadding(self, data, shape: Shape):
        width, height = shape.w, shape.h

        newWidth = shape.w + self.padding*2
        newHeight = shape.h + self.padding*2
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
            

    def forward(self, input, shape:Shape, active:string):
        if active not in ['softmax', 'relu']:
            print("Error: Activation function not active")
            return
        output = np.zeros((self.numKernel, shape.w * shape.h))

        for ilayer in range(self.numKernel):
            for ir in range(self.padding, self.padding + shape.h):
                for ic in range(self.padding, self.padding+ shape.w):
                    value = 0
                    for irk in range(self.sizeKernel.h):
                        for ick in range(self.sizeKernel.w):
                            r = ir - self.padding + irk
                            c = ic - self.padding + ick
                            value += input[ilayer][r*(shape.w+ self.padding*2) + c] * self.weight[ilayer][irk * self.sizeKernel.w + ick]
                    output[ilayer][(ir - self.padding)* shape.w + (ic - self.padding)] = value + self.bias[ilayer]
        return output



