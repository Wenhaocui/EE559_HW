import numpy as np
from sklearn.preprocessing import StandardScaler
import random

training = np.loadtxt('wine_train.csv',delimiter =',')
testing = np.loadtxt('wine_test.csv',delimiter =',')
scaler = StandardScaler()
scaler_t = StandardScaler()
training_label = np.array(training[:,13])
training_data = np.array(training[:,:13])
training_data_1st_two = np.array(training[:,:2])
training_data_new = scaler.fit_transform(training_data)
print(scaler.mean_)
print(np.sqrt(scaler.var_))
training_data_new_1st_two = scaler.fit_transform(training_data_1st_two)


testing_label = np.array(testing[:,13])
testing_data = np.array(testing[:,:13])
testing_data_1st_two = np.array(testing[:,:2])
testing_data_new = scaler_t.fit_transform(testing_data)
testing_data_new_1st_two = scaler_t.fit_transform(testing_data_1st_two)



from sklearn.linear_model import Perceptron
def Percep(training_data, testing_data, f, coef_init, intercept_init):
    ppn = Perceptron(warm_start = True)
    ppn.fit(training_data, training_label)
    ppn._allocate_parameter_mem(n_classes = 3, n_features = f, coef_init = coef_init, intercept_init = intercept_init)
    ppn.fit(training_data, training_label)
    weight = np.c_[(ppn.coef_) ,(ppn.intercept_).T]
    acc_tra = ppn.score(training_data,training_label)
    acc_tes = ppn.score(testing_data,testing_label)
    result = (weight,acc_tra,acc_tes)
    return result

coef_init = np.zeros((3,2))
intercept_init = np.zeros(3)
ppn_2 = Percep(training_data_new_1st_two, testing_data_new_1st_two, 2, coef_init, intercept_init)
print("Its weight is : %s" %ppn_2[0])
print("Accuracy Score of training data in 1st 2 feature: %f" % ppn_2[1])
print("Accuracy Score of testing data in 1st 2 feature: %f" % ppn_2[2])

coef_init = np.zeros((3,13))
intercept_init = np.zeros(3)
ppn_13 = Percep(training_data_new, testing_data_new, 13, coef_init, intercept_init)
print("Its weight is : %s" %ppn_13[0])
print("Accuracy Score of training data in all features: %f" % ppn_13[1])
print("Accuracy Score of testing data in all features: %f" % ppn_13[2])



for i in range(100):
    coef_init = np.random.rand(3,2)*100
    intercept_init = np.random.rand(3)*100
    result  = (0,0,0)
    ppn_2_max = Percep(training_data_new_1st_two, testing_data_new_1st_two, 2, coef_init, intercept_init)
    if ppn_2_max[1] >= result[1]:
        result = ppn_2_max
print("Its weight is : %s" %result[0])
print("Accuracy Score of training data 1st 2 feature MAX in 100: %f" % result[1])
print("Accuracy Score of testing data 1st 2 feature MAX in 100: %f" % result[2])


for i in range(100):
    coef_init = np.random.rand(3,13)*100
    intercept_init = np.random.rand(3)*100
    result  = (0,0,0)
    ppn_13_max = Percep(training_data_new, testing_data_new, 13, coef_init, intercept_init)
    if ppn_13_max[1] >= result[1]:
        result = ppn_13_max
print("Its weight is : %s" %result[0])
print("Accuracy Score of training data in all features MAX in 100: %f" % result[1])
print("Accuracy Score of testing data in all features MAX in 100: %f" % result[2])

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class MSE_binary(LinearRegression):
    def __init__(self):
        # print('Calling newly created MSE binary function...')
        super(MSE_binary, self).__init__()
    def predict(self,X):
        thr = 0.5
        y = self._decision_function(X)
        y_new = np.zeros(y.shape)
        y_new[y > thr] = 1
        return y_new

from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
binary_model = MSE_binary()
mc_model = OneVsRestClassifier(binary_model)
mc_model.fit(training_data_1st_two, training_label)
pred = mc_model.predict(testing_data_1st_two)
acc = accuracy_score(pred, testing_label)
print("Accuracy Score of testing data in 1st 2 feature unstandardized: %f" % acc)

binary_model = MSE_binary()
mc_model = OneVsRestClassifier(binary_model)
mc_model.fit(training_data, training_label)
pred = mc_model.predict(testing_data)
acc = accuracy_score(pred, testing_label)
print("Accuracy Score of testing data in all features unstandardized: %f" % acc)

binary_model = MSE_binary()
mc_model = OneVsRestClassifier(binary_model)
mc_model.fit(training_data_new_1st_two, training_label)
pred = mc_model.predict(testing_data_new_1st_two)
acc = accuracy_score(pred, testing_label)
print("Accuracy Score of testing data in 1st 2 features standardized: %f" % acc)

binary_model = MSE_binary()
mc_model = OneVsRestClassifier(binary_model)
mc_model.fit(training_data_new, training_label)
pred = mc_model.predict(testing_data_new)
acc = accuracy_score(pred, testing_label)
print("Accuracy Score of testing data in all features standardized: %f" % acc)

