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

#________________________________________________________________________________________________________________________

def termination(JL):
    if len(JL)>2 and np.abs(JL[-1]-JL[-2])<10**-8:
        return True
    else:
        return False
    
  
#running different architectures
def autotrain(nnList, xtr, ytr, g_eta, epochs, archlabel, xte=None, yte=None ):
    JLtr = []
    JLte = []
    epochL = []
    for e in tqdm(range(1,epochs+1)):
        p = np.random.permutation(len(xtr))
        xTrain = xtr[p]
        yTrain = ytr[p]

        h = xTrain
        for i in range(len(nnList)-1):
            h = nnList[i].forward(h)
        
        
        Jtr = nnList[-1].eval(yTrain, h)


        #----------------------------------------------------------------------------------------------------------------------

        # Backward pass 
        grad = nnList[-1].gradient(yTrain, h)
        for i in range(len(nnList)-2, -1, -1):
            newgrad = nnList[i].backward(grad)
            if isinstance(nnList[i], FullyConnectedLayer):
                nnList[i].updateWeights(grad, g_eta, e)
            grad = newgrad
            
        
        #print(total_mb)
        #----------------------------------------------------------------------------------------------------------------------
        #Forward pass through training set
        tr = xtr
        for i in range(len(nnList)-1):
            tr = nnList[i].forward(tr)
        Jtr = nnList[-1].eval(ytr, tr)        


        JLtr.append(Jtr)

        #----------------------------------------------------------------------------------------------------------------------
        #VALIDATION SET:
        if xte is not None:
            te = xte
            for i in range(len(nnList)-1):
                te = nnList[i].forward(te)
            Jte = nnList[-1].eval(yte, te)
            JLte.append(Jte)

            print(Jtr, Jte)
        else:
            print(Jtr)

        epochL.append(e)
        if termination(JLtr):
            break
#----------------------------------------------------------------------------------------------------------------------
    if xte is not None:
        #Accuracy - Training
        atr = 0
        predictiontr = np.argmax(tr, axis=1).reshape(tr.shape[0],1)
        predictionytr = np.argmax(ytr, axis=1).reshape(ytr.shape[0],1)
        predictionyte = np.argmax(yte, axis=1).reshape(yte.shape[0],1)


        for i in range(predictiontr.shape[0]):
            if predictiontr[i,:] == predictionytr[i,:]:
                atr += 1
        atr = atr*100/predictiontr.shape[0]
        print("Training Accuracy: ", atr)

        #Accuracy - Validation
        ate = 0
        predictionte = np.argmax(te, axis=1).reshape(te.shape[0],1)
        for i in range(predictionte.shape[0]):
            if predictionte[i,:] == predictionyte[i,:]:
                ate += 1
        ate = ate*100/predictionte.shape[0]
        print("Validation Accuracy: ", ate)


    #Plot J vs epochs for training and validation set
    plt.plot(epochL, JLtr, c='darkblue', label='Training J vs Epochs')
    if xte is not None:
        plt.plot(epochL, JLte, c='lawngreen', label='Validation J vs Epochs')
    plt.xlabel("Epochs")

    plt.ylabel("J")
    plt.legend()
    if xte is not None:
        plt.title(archlabel+f': Training Accuracy - {atr}, Validation Accuracy - {ate}')
    else:
        plt.title(archlabel)
    plt.show()


if __name__=="__main__":
    np.random.seed(0)

    #THIS FILE FOCUSES ON TRAINING WITHOUT MINIBATCHES. 
#_________________________
    X = np.load('./X.npy')
    Y =  np.load('./Y.npy')
    Y = Y.reshape(Y.shape[0],1)
    #Y_train = np.full((X_train.shape[0],1),1,dtype =int)
#_________________________

    #shuffling the dataset. and splitting into training and test set(2/3-Training, 1/3-Test)
    p = np.random.permutation(X.shape[0])
    X_train, X_test = X[p][:2*X.shape[0]//3,:], X[p][2*X.shape[0]//3:,:]
    Y_train, Y_test = Y[p][:2*Y.shape[0]//3], Y[p][2*Y.shape[0]//3:]

    #One hot encoding for 10 classes
    Y_train = np.squeeze(np.eye(10)[Y_train.astype(int).reshape(-1)])
    Y_test = np.squeeze(np.eye(10)[Y_test.astype(int).reshape(-1)])

    # print(X.shape, Y.shape)
    # print(X_train.shape,Y_train.shape)
    # print(X_test.shape, Y_test.shape)

#------------------------------------------------------BOX1--------------------------------------------------------
    #uncomment lines in BOX1 only to figure out what underlined size in FCL(__,10) should be initialised with. As many times we can forward pass through different number of convolutional maxpooling layers to figure out the size.  
    # cl1 = ConvolutionLayer(5)
    # y = cl1.forward(X_train)
    # ml = MaxpoolLayer(5,2)
    # y = ml.forward(y)
    # fl = FlattenLayer()
    # z = fl.forward(y)
    # print(z.shape)

    # cl2 = ConvolutionLayer(3)
    # y = cl2.forward(y)
    # ml2 = MaxpoolLayer(3,1)
    # z = ml2.forward(y)
    # # fl = FlattenLayer()
    # # z = fl.forward(z)
    # print(z.shape)
#------------------------------------------------------BOX1 END--------------------------------------------------------

    #WHILE RUNNING DIFFERENT MODELS UNCOMMENT THE model list and autotrain function following it. Comment out other models.

    # MODEL 1 - Change FCL weight initialisations in range(-0.067,0.067)
    nn1 = [ConvolutionLayer(5), MaxpoolLayer(3,2), FlattenLayer(), FullyConnectedLayer(17*17,50), ReLuLayer(), FullyConnectedLayer(50,10), SoftmaxLayer(), CrossEntropy()]
    autotrain(nn1, X_train, Y_train, 0.05, 75, "nn1", X_test, Y_test)

    # # MODEL 2 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn2 = [ConvolutionLayer(5), MaxpoolLayer(3,2), FlattenLayer(), FullyConnectedLayer(17*17,50), ReLuLayer(), FullyConnectedLayer(50,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn2, X_train, Y_train, 0.017, 75, "nn2", X_test, Y_test)
    
    # # MODEL 3 - in code b.py

    # # MODEL 4 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn4 = [ConvolutionLayer(5), MaxpoolLayer(5,2), FlattenLayer(), FullyConnectedLayer(16*16,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn4, X_train, Y_train, 0.05, 50, "nn4", X_test, Y_test)

    # # MODEL 5 - in code b.py

    # # MODEL 6 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn6 = [ConvolutionLayer(5), MaxpoolLayer(5,2), FlattenLayer(), FullyConnectedLayer(16*16,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn6, X_train, Y_train, 0.03, 75, "nn6", X_test, Y_test)

    # # MODEL 7 - Change FCL weight initialisations in range(-0.1,0.1)
    # nn7 = [ConvolutionLayer(5), MaxpoolLayer(5,2), FlattenLayer(), FullyConnectedLayer(16*16,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn7, X_train, Y_train, 0.1, 75, "nn6", X_test, Y_test)

    # # MODEL 8 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn8 = [ConvolutionLayer(5), MaxpoolLayer(5,2), FlattenLayer(), FullyConnectedLayer(16*16,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn8, X_train, Y_train, 0.03, 75, "nn8", X_test, Y_test)

    
    # # MODEL 9 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn9 = [ConvolutionLayer(5), MaxpoolLayer(5,2), FlattenLayer(), FullyConnectedLayer(16*16,10), SigmoidLayer(), LogLoss()]
    # autotrain(nn9, X_train, Y_train, 0.05, 75, "nn9", X_test, Y_test)

    # # MODEL 10 - Change FCL weight initialisations in range(-0.067,0.067)
    # nn10 = [ConvolutionLayer(3), MaxpoolLayer(5,2), ConvolutionLayer(5), MaxpoolLayer(3,2), FlattenLayer(), FullyConnectedLayer(49,10), SoftmaxLayer(), CrossEntropy()]
    # autotrain(nn10, X_train, Y_train, 0.05, 75, "nn10", X_test, Y_test)

    # # MODEL 11 - in code b.py
