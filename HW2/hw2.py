import numpy as np
import math
import plotDecBoundaries2
import plotDecBoundaries3

training1 = np.loadtxt('synthetic1_train.csv',delimiter =',')
training2 = np.loadtxt('synthetic2_train.csv',delimiter =',')
testing1 = np.loadtxt('synthetic1_test.csv',delimiter =',')
testing2 = np.loadtxt('synthetic2_test.csv',delimiter =',')

def errorRate(testing, s1, s2, s3, fea1, fea2, cla):
    correct = 0
    for i in range(len(testing)):
        dist_1 = math.sqrt( (testing[i][fea1]- s1[0][0])**2 + (testing[i][fea2]- s1[0][1])**2)
        dist_2 = math.sqrt( (testing[i][fea1]- s1[1][0])**2 + (testing[i][fea2]- s1[1][1])**2)
        dist_3 = math.sqrt( (testing[i][fea1]- s2[0][0])**2 + (testing[i][fea2]- s2[0][1])**2)
        dist_4 = math.sqrt( (testing[i][fea1]- s2[1][0])**2 + (testing[i][fea2]- s2[1][1])**2)
        dist_5 = math.sqrt( (testing[i][fea1]- s3[0][0])**2 + (testing[i][fea2]- s3[0][1])**2)
        dist_6 = math.sqrt( (testing[i][fea1]- s3[1][0])**2 + (testing[i][fea2]- s3[1][1])**2)
        if ((min(dist_1, dist_2) == dist_1 and min(dist_3, dist_4) == dist_4 and min(dist_5, dist_6) == dist_6 and testing[i][cla] == 1)
        or (min(dist_1, dist_2) == dist_2 and min(dist_3, dist_4) == dist_3 and min(dist_5, dist_6) == dist_6 and testing[i][cla] == 2) 
        or (min(dist_1, dist_2) == dist_2 and min(dist_3, dist_4) == dist_4 and min(dist_5, dist_6) == dist_5 and testing[i][cla] == 3)):
            correct += 1
    rate = correct/len(testing)
    return rate

def sampleMean(training, c, fea1, fea2, cla):
    class1_x = 0
    class1_y = 0
    class2_x = 0
    class2_y = 0
    num1 = 0
    num2 = 0
    for i in training:
        if (i[cla] == c):
            class1_x += i[fea1]
            class1_y += i[fea2]
            num1 += 1
        else:
            class2_x += i[fea1]
            class2_y += i[fea2]
            num2 += 1
    sample_mean = np.array([[class1_x/num1, class1_y/num1],[class2_x/num2, class2_y/num2]])
    return(sample_mean)

 
training3 = np.loadtxt('wine_train.csv',delimiter =',')
testing3 = np.loadtxt('wine_test.csv',delimiter =',')
s1 = sampleMean(training3, 1, 0, 1, 13)
s2 = sampleMean(training3, 2, 0, 1, 13)
s3 = sampleMean(training3, 3, 0, 1, 13)


print(errorRate(training3, s1, s2, s3, 0, 1, 13))
print(errorRate(testing3, s1, s2, s3, 0, 1, 13))
plotDecBoundaries2.plotDecBoundaries(training3, training3[:,13], s1, 1)
plotDecBoundaries2.plotDecBoundaries(training3, training3[:,13], s2, 2)
plotDecBoundaries2.plotDecBoundaries(training3, training3[:,13], s3, 3)

plotDecBoundaries3.plotDecBoundaries(training3, training3[:,13], s1, s2, s3)