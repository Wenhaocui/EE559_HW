import numpy as np
import math
import plotDecBoundaries

training1 = np.loadtxt('synthetic1_train.csv',delimiter =',')
training2 = np.loadtxt('synthetic2_train.csv',delimiter =',')
testing1 = np.loadtxt('synthetic1_test.csv',delimiter =',')
testing2 = np.loadtxt('synthetic2_test.csv',delimiter =',')

def errorRate(testing, sample_mean, fea1, fea2, cla):
    error_1 = 0
    for i in range(len(testing)):
        dist_1 = math.sqrt( (testing[i][fea1]- sample_mean[0][0])**2 + (testing[i][fea2]- sample_mean[0][1])**2)
        dist_2 = math.sqrt( (testing[i][fea1]- sample_mean[1][0])**2 + (testing[i][fea2]- sample_mean[1][1])**2)
        dist_3 = 0
        if len(sample_mean) == 3:
            dist_3 = math.sqrt( (testing[i][fea1]- sample_mean[2][0])**2 + (testing[i][fea2]- sample_mean[2][1])**2)
        if (dist_3 == 0):
            if (min(dist_1, dist_2) == dist_1 and testing[i][cla] != 1) or (min(dist_1, dist_2) == dist_2 and testing[i][cla] != 2):
                error_1 += 1
        else:
            if (min(dist_1, dist_2, dist_3) == dist_1 and testing[i][cla] != 1) or (min(dist_1, dist_2, dist_3) == dist_2 and testing[i][cla] != 2) or (min(dist_1, dist_2, dist_3) == dist_3 and testing[i][cla] != 3):
                error_1 += 1
    rate_1 = error_1/len(testing)
    return rate_1

def sampleMean(training, testing, fea1, fea2, cla):
    class1_x = 0
    class1_y = 0
    class2_x = 0
    class2_y = 0
    class3_x = 0
    class3_y = 0
    num1 = 0
    num2 = 0
    num3 = 0
    if (max(training[:,cla]) == 3):
        for i in training:
            if (i[cla] == 1):
                class1_x += i[fea1]
                class1_y += i[fea2]
                num1 += 1
            elif (i[cla] == 2):
                class2_x += i[fea1]
                class2_y += i[fea2]
                num2 += 1
            elif (i[cla] == 3):
                class3_x += i[fea1]
                class3_y += i[fea2]
                num3 += 1
        sample_mean = np.array([[class1_x/num1, class1_y/num1],[class2_x/num2, class2_y/num2],[class3_x/num3, class3_y/num3]])
    elif (max(training[:,cla]) == 2):
        for i in training:
            if (i[cla] == 1):
                class1_x += i[fea1]
                class1_y += i[fea2]
                num1 += 1
            elif (i[cla] == 2):
                class2_x += i[fea1]
                class2_y += i[fea2]
                num2 += 1
        sample_mean = np.array([[class1_x/num1, class1_y/num1],[class2_x/num2, class2_y/num2]])
    return(sample_mean)

#   Question a
s1 = sampleMean(training1, testing1, 0, 1, 2)
print(errorRate(training1, s1, 0, 1, 2))
print(errorRate(testing1, s1, 0, 1, 2))
plotDecBoundaries.plotDecBoundaries(training1, training1[:,2], s1)
s2 = sampleMean(training2, testing2, 0, 1, 2)
print(errorRate(training2, s2, 0, 1, 2))
print(errorRate(testing2, s2, 0, 1, 2))
plotDecBoundaries.plotDecBoundaries(training2, training2[:,2], s2)

#   Question c
training3 = np.loadtxt('wine_train.csv',delimiter =',')
testing3 = np.loadtxt('wine_test.csv',delimiter =',')
s3 = sampleMean(training3, testing3, 0, 1, 13)
print(errorRate(training3, s3, 0, 1, 13))
print(errorRate(testing3, s3, 0, 1, 13))
plotDecBoundaries.plotDecBoundaries(training3, training3[:,13], s3)

#   Question d
set_val1 = 1
set_val2 = 1
f1 = 0
f2 = 0
f3 = 0
f4 = 0
s6 = 0
s7 = 0
its_tes = 0

for i in range(len(training3[0])-1):
    for j in range(i+1,len(training3[0])-1):
        s5 = sampleMean(training3, testing3, i, j, 13)
        tra = errorRate(training3, s5, i, j, 13)
        tes = errorRate(testing3, s5, i, j, 13)
        if (tra < set_val1):
                set_val1 = tra
                f1 = i
                f2 = j
                s6 = s5
                its_tes = errorRate(testing3, s6, i, j, 13)
        if (tes < set_val2):
                set_val2 = tes
                f3 = i
                f4 = j
                s7 =s5
                its_tra = errorRate(training3, s6, i, j, 13)
print ('The smallest training error rate is %s appearing at feature %s and feature %s, its testing error rate is %s' % (set_val1, f1+1, f2+1, its_tes))
plotDecBoundaries.plotDecBoundaries(training3, training3[:,13], s6)
print ('The smallest testing error rate is %s appearing at feature %s and feature %s, its training error rate is %s' % (set_val2, f3+1, f4+1, its_tra))
plotDecBoundaries.plotDecBoundaries(training3, training3[:,13], s7)