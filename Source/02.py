import sys
import numpy as np
from abc import ABC, abstractmethod 
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import cv2
import math
import scipy.signal

np.set_printoptions(threshold=sys.maxsize)

class Layer(ABC):
    def __init__(self):
        self.prevIn = []
        self.prevOut = []
        self.w = []
        self.a = 0.66

    def setPrevIn(self ,dataIn): 
        self.prevIn = dataIn

    def setPrevOut( self , out ): 
        self.prevOut = out
        
    def getPrevIn(self): 
        return self.prevIn 

    def getPrevOut(self): 
        return self.prevOut

    def backward(self, gradIn): 
        pass

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass
  

class InputLayer(Layer):
    def __init__(self, dataIn):
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, ddof=1, axis=0)
        self.stdX[self.stdX==0] = 1

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored
    
    def gradient(self):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self ,sizeIn ,sizeOut):
        self.weight = np.random.uniform(low=-0.067, high=0.067, size=(sizeIn,sizeOut))
        self.bias = np.random.uniform(low=-0.067, high=0.067, size=(1,sizeOut))
         self.sw = 0
        self.rw = 0
        self.sb = 0
        self.rb = 0
        self.w = self.weight

    def getWeights(self):
        return self.weight

    def setWeights(self , weights):
        self.weight = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias): 
        self.bias = bias

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.dot(dataIn, self.weight) + self.bias
        self.setPrevOut(y)
        return y

    def gradient(self): 
        tensor = self.getWeights()
        return tensor.T

    def backward(self, gradIn): 
        fclGrad = np.dot(gradIn, self.gradient())
        return fclGrad

    def updateWeights(self, gradIn, eta, epochCnt, d1=0.9, d2=0.999, ns=10e-8):
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        dJdw = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
        self.sw = d1*self.sw + (1-d1)*dJdw
        self.rw = d2*self.rw + (1-d2)*np.square(dJdw)
        self.sb = d1*self.sb + (1-d1)*dJdb
        self.rb = d2*self.rb + (1- d2)*np.square(dJdb)

        self.setWeights(self.weight - eta*(self.sw/(1-d1**epochCnt))/(np.sqrt(self.rw/(1-d2**epochCnt))+ns))
        self.w = self.weight
        self.setBias(self.bias - eta*(self.sb/(1-d1**epochCnt))/(np.sqrt(self.rb/(1-d2**epochCnt))+ns))

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = dataIn
        self.setPrevOut(y)
        return y
    
    tensor method
    def gradient(self):
        i = np.identity(self.prevIn.shape[1])
        tensor = np.array([i] * self.prevIn.shape[0])
        return tensor

    def backward(self,gradIn):
        linearGrad = np.multiply(gradIn, self.gradient())
        return linearGrad
    
class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = np.maximum(0,dataIn)
        self.setPrevOut(y)
        return y
    
    Tensor method
    def gradient(self):
        i = np.identity(self.prevIn.shape[1])
        tensor = np.array([i] * self.prevIn.shape[0])
        tensor[self.prevIn<0]=0
        return tensor

    def backward(self,gradIn):
        reluGrad = np.multiply(gradIn, self.gradient())
        return reluGrad

        
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = 1/(1+np.exp(-dataIn))
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        tensor = np.ndarray(shape=(self.prevIn.shape[0],self.prevIn.shape[1],self.prevIn.shape[1]))
        tensor.fill(0.0)
        for r in range(self.prevIn.shape[0]):
            for c in range(self.prevIn.shape[1]):
                tensor[r][c][c] = self.prevOut[r][c]*(1-self.prevOut[r][c])+0.0000001
        return tensor
    
    def backward(self,gradIn):
        sigmoidGrad = np.multiply(gradIn, self.gradient())
        return sigmoidGrad


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)) / np.sum(np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)), axis=1, keepdims=True)
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        y = self.prevOut
        n = y.shape[0]
        k = y.shape[1]
        tensor = np.ndarray(shape=(n,k,k))
        tensor.fill(0.0) 
        for l in range(n):
            for i in range(len(y[l])):
                for j in range(len(y[l])):
                    if i == j:
                        tensor[l][i][j] = y[l][i] * (1-y[l][i])
                    else:
                        tensor[l][i][j] = -y[l][i] * y[l][j]
        return tensor

    def backward(self,gradIn):
        softmaxGrad = np.zeros((gradIn.shape[0],self.gradient().shape[1]))
        #for each observation computation. 
        for n in range(gradIn.shape[0]): 
            softmaxGrad[n,:] = gradIn[n,:]@self.gradient()[n,:,:]
        return softmaxGrad


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = (np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)) - np.exp(-dataIn+np.min(dataIn,axis=1,keepdims=True)))/ (np.exp(dataIn-np.max(dataIn,axis=1,keepdims=True)) + np.exp(-dataIn+np.min(dataIn,axis=1,keepdims=True)) +0.000001)
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        tensor = np.ndarray(shape=(self.prevIn.shape[0],self.prevIn.shape[1],self.prevIn.shape[1]))
        tensor.fill(0.0)
        for r in range(self.prevIn.shape[0]):
            for c in range(self.prevIn.shape[1]):
                tensor[r][c][c] = (1-self.prevOut[r][c]*self.prevOut[r][c])+0.0000001
        return tensor

    def backward(self,gradIn):
        tanhGrad = np.multiply(gradIn, self.gradient())
        return tanhGrad

class DropoutLayer(Layer):
    def __init__(self, pr=0.5):
        self.pr = pr
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.mask = np.random.binomial(1,self.pr,size=dataIn.shape) / self.pr
        y = np.multiply(dataIn,self.mask)
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        pass

    def backward(self,gradIn):
        grad = np.multiply(gradIn, self.mask)
        return grad
    

class ConvolutionLayer(Layer):
    def __init__(self ,kernalSize=3):
        self.kernel = np.random.uniform(-0.1, 0.1, size=(kernalSize,kernalSize))
        self.sk = 0
        self.rk = 0

    def getKernel(self):
        return self.kernel

    def setKernel(self , kernel):
        self.kernel = kernel

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.zeros((dataIn.shape[0], dataIn.shape[1], dataIn.shape[2]))
        for i in range(dataIn.shape[0]):

            y[i,:,:] = scipy.signal.correlate2d(dataIn[i,:,:], self.kernel, mode='same')
        self.setPrevOut(y)
        return y

    def gradient(self):
        return self.kernel.T

    def backward(self,gradIn):
        convgrad = np.zeros((self.prevIn.shape[0], self.prevIn.shape[1], self.prevIn.shape[2]))
        for i in range(gradIn.shape[0]):
            padded = gradIn[i,:,:]
            convgrad[i,:,:] = scipy.signal.correlate2d(padded, self.gradient(), mode='same')
        return convgrad


    def updateWeights(self, gradIn, eta, epochCnt, d1=0.9, d2=0.999, ns=10e-8):
        djdk = np.zeros((gradIn.shape[0], gradIn.shape[1], gradIn.shape[2]))
        for i in range(gradIn.shape[0]):
            djdk[i,:,:] = scipy.signal.correlate2d(gradIn[i,:,:], self.getPrevIn(), mode='valid')
            self.setKernel(self.kernel - eta*djdk[i,:,:])

class MaxpoolLayer(Layer):
    def __init__(self,pool_size=3,stride = 3):
        super().__init__()
        self.poolSize = pool_size
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.dataIn = dataIn

        obs , inW, inH= dataIn.shape
        h = int((inH - self.poolSize) / self.stride) + 1     
        w = int((inW - self.poolSize) / self.stride) + 1
        out = np.zeros((obs,w,h))
                
        for i in range(obs):                      
            currY = 0
            outY = 0                             
            while currY + self.poolSize <= inH:  # slide the max pooling window vertically across the image
                currX = 0
                outX = 0
                while currX + self.poolSize <= inW:   # slide the max pooling window horizontally across the image
                    arr_area = dataIn[i, currX:currX + self.poolSize, currY:currY + self.poolSize]
                    out[i, outX, outY] = np.max(arr_area)    # choose the maximum value within the window
                    currX += self.stride                       
                    outX += 1
                currY += self.stride
                outY += 1

        self.setPrevOut(out)
        return out
    
    def gradient(self):
        pass

    def backward(self,gradIn):
        obs, w, h= self.getPrevIn().shape                                                                 
        grad = np.zeros((obs, w, h))

        for c in range(obs):
            flagY = 0
            outY = 0
            while flagY + self.poolSize <= h:
                flagX = 0
                outX = 0
                while flagX + self.poolSize <= w:
                    patch = self.dataIn[c, flagX:flagX + self.poolSize,flagY:flagY + self.poolSize]  
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    grad[c, flagX + x,flagY + y] = gradIn[c, outX,outY]#check this line
                    flagX += self.stride
                    outX += 1
                flagY += self.stride
                outY += 1
        return grad

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        y = np.zeros((dataIn.shape[0], dataIn.shape[1]*dataIn.shape[2]))
        for i in range(dataIn.shape[0]):
            #row major indexing
            y[i,:] = dataIn[i,:,:].flatten()
        self.setPrevOut(y)
        return y
    def gradient(self):
        pass

    def backward(self,gradIn):
        z = np.zeros((self.prevIn.shape[0], self.prevIn.shape[1], self.prevIn.shape[2]))
        for j in range(gradIn.shape[0]):
            z[j,:,:] = gradIn[j,:].reshape(self.prevIn.shape[1], self.prevIn.shape[2])
        return z


class SquaredError(Layer):
    def eval(self ,Y, Yhat): 
        return np.sum(np.square(Y-Yhat))/Y.shape[0]

    def gradient(self ,Y, Yhat):     
        return -2*(Y-Yhat)

class LogLoss(Layer):
    def eval(self ,Y, Yhat): 
        return np.mean(-(np.multiply(Y,np.log(Yhat + 0.0000001)) + (np.multiply((1-Y),np.log(1-Yhat + 0.0000001)))))
    def gradient(self ,Y, Yhat):     
        return -((Y - Yhat)/(Yhat*(1-Yhat) + 0.0000001))


class CrossEntropy(Layer):
    def __init__(self):
        super().__init__()
    def forward(self,dataIn):
        pass
    def eval(self ,Y, Yhat): 
        r = np.zeros((Y.shape[0],Y.shape[1]))
        r.fill(1-self.a)
        return -np.sum(np.multiply(np.multiply(Y, (np.log(Yhat+0.0000001))),r))/ Y.shape[0]-self.a*(np.sum(np.multiply(self.w,self.w)))
    def gradient(self ,Y, Yhat): 
        r = np.zeros((Y.shape[0],Y.shape[1]))
        r.fill(1-self.a)
        return np.multiply(-(Y/(Yhat+0.0000001)),r)-2*self.a*np.sum(self.w)


if __name__=="__main__":
    np.random.seed(0)

    #This is to save input and label matrix representations for all the obervations(given images) into a .npy file that can be loaded directly. 
    dataPath = "./10classpins/"
    labels = glob.glob(dataPath+"*/", recursive = True)
    arr = []
    lb = []
    for i in tqdm(range(len(labels))):
        images = sorted(glob.glob(labels[i]+'resizedgray/'+'*.jpg'))
        for j in tqdm(range(len(images))):
            img = cv2.imread(images[j], cv2.IMREAD_GRAYSCALE)
            #image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
            #image_padded[1:-1, 1:-1] = img
            #f = image_padded.flatten()
            #imfrow = f.reshape(1,f.shape[0])
            arr.append(img)
            lb.append(i)

    X = np.array(arr)/255
    np.save("X", X)
    print(X.shape)
    
    
    Y = np.array(lb)
    Y = Y.reshape(Y.shape[0],1)
    np.save("Y", Y)
    print(Y.shape)
