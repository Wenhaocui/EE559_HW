import numpy as np
import math
import plotSVMBoundaries

training1_x = np.loadtxt('HW8_1_csv/train_x.csv',delimiter =',')
training1_y = np.loadtxt('HW8_1_csv/train_y.csv',delimiter =',')

# Q1(a)

from sklearn.svm import SVC
clf_1 = SVC(C=1,kernel='linear')
clf_1.fit(training1_x, training1_y)
pre_1 = clf_1.predict(training1_x)
print(clf_1.score(training1_x, training1_y))
plotSVMBoundaries.plotSVMBoundaries(training1_x, training1_y, clf_1)
clf_100 = SVC(C=100,kernel='linear')
clf_100.fit(training1_x, training1_y)
print(clf_100.score(training1_x, training1_y))
plotSVMBoundaries.plotSVMBoundaries(training1_x, training1_y, clf_100)

# Q1(b)
support_vectors = clf_100.support_vectors_
print(support_vectors)
print(clf_100.coef_)
print(clf_100.intercept_)
plotSVMBoundaries.plotSVMBoundaries(training1_x, training1_y, clf_100, support_vectors)

# Q1(c)
for i in range(len(support_vectors)):
    w = clf_100.coef_
    x = (support_vectors[i]).T
    g = np.dot(w,x)+ clf_100.intercept_
    print(g)

# Q1(d)
training2_x = np.loadtxt('HW8_2_csv/train_x.csv',delimiter =',')
training2_y = np.loadtxt('HW8_2_csv/train_y.csv',delimiter =',')

clf_50 = SVC(gamma='auto', C=50)
clf_50.fit(training2_x, training2_y)
print(clf_50.score(training2_x, training2_y))
plotSVMBoundaries.plotSVMBoundaries(training2_x, training2_y, clf_50, clf_50.support_vectors_)
clf_5000 = SVC(gamma='auto', C=5000)
clf_5000.fit(training2_x, training2_y)
print(clf_5000.score(training2_x, training2_y))
plotSVMBoundaries.plotSVMBoundaries(training2_x, training2_y, clf_5000, clf_5000.support_vectors_)

# Q1(e)
clf_g_10 = SVC(gamma=10)
clf_g_10.fit(training2_x, training2_y)
print(clf_g_10.score(training2_x, training2_y))
plotSVMBoundaries.plotSVMBoundaries(training2_x, training2_y, clf_g_10, clf_g_10.support_vectors_)

clf_g_50 = SVC(gamma=50)
clf_g_50.fit(training2_x, training2_y)
print(clf_g_50.score(training2_x, training2_y))
plotSVMBoundaries.plotSVMBoundaries(training2_x, training2_y, clf_g_50, clf_g_50.support_vectors_)

clf_g_500 = SVC(gamma=500)
clf_g_500.fit(training2_x, training2_y)
print(clf_g_500.score(training2_x, training2_y))
plotSVMBoundaries.plotSVMBoundaries(training2_x, training2_y, clf_g_500, clf_g_500.support_vectors_)


# Q2
feature_train = np.loadtxt('wine_csv/feature_train.csv',delimiter =',')
label_train = np.loadtxt('wine_csv/label_train.csv',delimiter =',')
feature_test = np.loadtxt('wine_csv/feature_test.csv',delimiter =',')
label_test = np.loadtxt('wine_csv/label_test.csv',delimiter =',')
feature_train = np.array(feature_train[:,:2])
feature_test  = np.array(feature_test[:,:2])

# Q2(a)
acc = 0
std = 0
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle= True)
# X_tr, X_v, y_tr, y_v = train_test_split(feature_train, label_train, test_size=(1/K))
for tr_index, v_index in skf.split(feature_train, label_train):
    X_tr, X_v = feature_train[tr_index], feature_train[v_index]
    y_tr, y_v = label_train[tr_index], label_train[v_index]
    clf = SVC()
    clf.fit(X_tr, y_tr)
    cur = clf.score(X_v, y_v)
    acc += cur
acc = acc/skf.get_n_splits(feature_train, label_train)
print(acc)

# Q2(b)
gam = np.logspace(-3,3,50)
c = np.logspace(-3,3,50)
ACC = np.zeros([50,50],dtype=float)
DEV = np.zeros([50,50],dtype=float)
for i in range(len(gam)):
    for j in range(len(c)):
        for tr_index, v_index in skf.split(feature_train, label_train):
            X_tr, X_v = feature_train[tr_index], feature_train[v_index]
            y_tr, y_v = label_train[tr_index], label_train[v_index]
            clf = SVC(gamma=gam[i], C=c[j])
            clf.fit(X_tr, y_tr)
            cur = clf.score(X_v, y_v)
            ACC[i][j] += cur
        ACC[i][j] = (ACC[i][j])/5
        for tr_index, v_index in skf.split(feature_train, label_train):
            X_tr, X_v = feature_train[tr_index], feature_train[v_index]
            y_tr, y_v = label_train[tr_index], label_train[v_index]
            clf = SVC(gamma=gam[i], C=c[j])
            clf.fit(X_tr, y_tr)
            cur = clf.score(X_v, y_v)
            DEV[i][j] += np.square(cur - ACC[i][j])
        DEV[i][j] = np.sqrt(DEV[i][j]/(skf.get_n_splits(feature_train, label_train)-1))
print(ACC)
print(DEV)

import matplotlib.pyplot as plt

plt.imshow(ACC,interpolation = 'nearest', cmap = 'Greens') 
plt.xticks([0,49],[0.001,1000])
plt.yticks([0,49],[0.001,1000])
plt.colorbar()
plt.show()


best_acc = np.amax(ACC)
print(best_acc)
best_acc_ind = np.argmax(ACC)
best_gam = gam[math.floor(best_acc_ind/50)]
best_C = c[best_acc_ind - math.floor(best_acc_ind/50)*50]
print(best_gam)
print(best_C)
print(DEV[math.floor(best_acc_ind/50)][best_acc_ind - math.floor(best_acc_ind/50)*50])


#Q2(c)
best_value = []
best_acc = 0
best_dev = 0
ACC_sum = np.zeros([50,50],dtype=float)
DEV_sum = np.zeros([50,50],dtype=float)
for T in range(20):
    ACC = np.zeros([50,50],dtype=float)
    DEV = np.zeros([50,50],dtype=float)
    for i in range(len(gam)):
        for j in range(len(c)):
            for tr_index, v_index in skf.split(feature_train, label_train):
                X_tr, X_v = feature_train[tr_index], feature_train[v_index]
                y_tr, y_v = label_train[tr_index], label_train[v_index]
                clf = SVC(gamma=gam[i], C=c[j])
                clf.fit(X_tr, y_tr)
                cur = clf.score(X_v, y_v)
                ACC[i][j] += cur
            ACC[i][j] = (ACC[i][j])/5
            for tr_index, v_index in skf.split(feature_train, label_train):
                X_tr, X_v = feature_train[tr_index], feature_train[v_index]
                y_tr, y_v = label_train[tr_index], label_train[v_index]
                clf = SVC(gamma=gam[i], C=c[j])
                clf.fit(X_tr, y_tr)
                cur = clf.score(X_v, y_v)
                DEV[i][j] += np.square(cur - ACC[i][j])
            DEV[i][j] = np.sqrt(DEV[i][j]/(skf.get_n_splits(feature_train, label_train)-1))
    best_acc_ind = np.argmax(ACC)
    best_gam = gam[math.floor(best_acc_ind/50)]
    best_C = c[best_acc_ind - math.floor(best_acc_ind/50)*50]
    best_value.append([best_gam,best_C])
    ACC_sum += ACC
    DEV_sum += DEV
    # print(ACC_sum[0][0])
    # # print(DEV_sum)
ACC_sum = ACC_sum/20
DEV_sum = DEV_sum/20
best_acc_sum_ind = np.argmax(ACC_sum)
best_sum_gam = gam[math.floor(best_acc_sum_ind/50)]
best_sum_C = c[best_acc_sum_ind - math.floor(best_acc_sum_ind/50)*50]
best_sum_value = [best_sum_gam,best_sum_C]
print(best_value)
print(best_sum_value)
best_acc_sum = np.amax(ACC_sum)
print(best_acc_sum)
print(DEV_sum[math.floor(best_acc_sum_ind/50)][best_acc_sum_ind - math.floor(best_acc_sum_ind/50)*50])

#Q2(d)
clf_fin = SVC(gamma=best_sum_gam,C=best_sum_C)
clf_fin.fit(feature_train, label_train)
print(clf_fin.score(feature_test, label_test))