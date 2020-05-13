################################################
## EE559 HW Wk2, Prof. Jenkins, Spring 2018
## Created by Arindam Jati, TA
## Tested in Python 3.6.3, OSX El Captain
################################################

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries(training, label_train, s1, s2, s3):

    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    # nclass =  max(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    dist_mat1 = cdist(xy, s1)
    dist_mat2 = cdist(xy, s2)
    dist_mat3 = cdist(xy, s3)
    pred_label = ((np.argmax(dist_mat1, axis=1) & np.argmin(dist_mat2, axis=1) & np.argmin(dist_mat3, axis=1)) 
    | 2*(np.argmax(dist_mat2, axis=1) & np.argmin(dist_mat1, axis=1) & np.argmin(dist_mat3, axis=1)) 
    | 3*(np.argmax(dist_mat3, axis=1) & np.argmin(dist_mat1, axis=1) & np.argmin(dist_mat2, axis=1)))
    
    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')
    plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'go')
    plt.plot(training[label_train == 3, 0],training[label_train == 3, 1], 'b*')


    l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    plt.gca().add_artist(l)
    
    m1, = plt.plot(s1[0,0], s1[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(s2[0,0], s2[0,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    m3, = plt.plot(s3[0,0], s3[0,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    # include legend for class mean vector
    l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    
    plt.gca().add_artist(l1)

    plt.show()


