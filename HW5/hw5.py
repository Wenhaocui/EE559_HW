import numpy as np
import math
import random 
import plotDecBoundaries4

training1 = np.loadtxt('synthetic1_train.csv',delimiter =',')
training2 = np.loadtxt('synthetic2_train.csv',delimiter =',')
training3_t = np.loadtxt('feature_train.csv',delimiter =',')
training3_l = np.loadtxt('label_train.csv',delimiter =',')


X1_f = training1[:,0:2]
cla1_f = training1[:,2]
X2_f = training2[:,0:2]
cla2_f = training2[:,2]
X3_f = training3_t
cla3_f = training3_l

def training(X,cla):
    w_ = np.ones(X.shape[1]+1)*0.1
    x_ =np.random.permutation(np.c_[X,np.ones(X.shape[0]),cla])

    # reflect
    for i in range(len(cla)):
        if x_[i][3] == 1.0:
            x_[i] = x_[i]
        elif x_[i][3] == 2.0:
            x_[i][0:3] = -(x_[i][0:3])

    # iteration
    n_iter = 1000
    eta = 1
    error = 0 
    epoch = 0    
    for _ in range(n_iter):
        for i in range(len(x_)):
            if np.dot(x_[i][0:3],w_)>0:
                w_ = w_
                
            elif np.dot(x_[i][0:3],w_)<=0:
                w_ += eta * x_[i][0:3]
                error += 1
        epoch += 1
        if(error > 0 and epoch<1000):
            error = 0
            
            pass
        else:
            break
    
    print('The error rate is %s' % (error/X.shape[0]))
    print(w_)
    return(w_)


testing1 = np.loadtxt('synthetic1_test.csv',delimiter =',')
testing2 = np.loadtxt('synthetic2_test.csv',delimiter =',')
testing3_t = np.loadtxt('feature_test.csv',delimiter =',')
testing3_l = np.loadtxt('label_test.csv',delimiter =',')

X1_t = testing1[:,0:2]
cla1_t = testing1[:,2]
X2_t = testing2[:,0:2]
cla2_t = testing2[:,2]
X3_t = testing3_t
cla3_t = testing3_l
w1 = training(X1_f,cla1_f)
w2 = training(X2_f,cla2_f)
w3 = training(X3_f,cla3_f)

def testing(X,cla,w_):
    x_ =np.random.permutation(np.c_[X,np.ones(X.shape[0]),cla])
    n_iter = 1000
    error = 0 
    epoch = 0  

    for _ in range(n_iter):
        for i in range(len(x_)):
            if (np.dot(x_[i][0:3],w_)>0 and x_[i][3] == 1.0) or (np.dot(x_[i][0:3],w_)<=0 and x_[i][3] == 2.0):
                pass
            elif (np.dot(x_[i][0:3],w_)>0 and x_[i][3] == 2.0) or (np.dot(x_[i][0:3],w_)<=0 and x_[i][3] == 1.0):
                error += 1
        epoch += 1
        if(error > 0 and epoch<1000):
            error = 0
            
            pass
        else:
            break
    return(error/X.shape[0])


t1 = testing(X1_t,cla1_t,w1)
t2 = testing(X2_t,cla2_t,w2)
t3 = testing(X3_t,cla3_t,w3)
print('The testing1 error rate is %s' % (t1))
print('The testing2 error rate is %s' % (t2))
print('The testing3 error rate is %s' % (t3))
plotDecBoundaries4.plotDecBoundaries(X1_f, cla1_f, w1)
plotDecBoundaries4.plotDecBoundaries(X1_t, cla1_t, w1)
plotDecBoundaries4.plotDecBoundaries(X2_f, cla2_f, w2)
plotDecBoundaries4.plotDecBoundaries(X2_t, cla2_t, w2)
plotDecBoundaries4.plotDecBoundaries(X3_f, cla3_f, w3)
plotDecBoundaries4.plotDecBoundaries(X3_t, cla3_t, w3)