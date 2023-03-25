import scipy.io as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


data = sp.loadmat('./dataset.mat')

x_train = [[x for x in train_arr_x] for train_arr_x in data['X_trn']] 
y_train_arr = [[y for y in train_arr_y] for train_arr_y in data['Y_trn']]
y_train = [y[0] for y in y_train_arr]

x_test = [[x for x in test_arr_x] for test_arr_x in data['X_tst']]
y_test_arr = [[y for y in test_arr_y] for test_arr_y in data['Y_tst']] 
y_test = [y[0] for y in y_test_arr]

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

results = classifier.predict(x_test)
num_preds = len(y_test)

x_coord_1 = []
y_coord_1 = []

x_coord_0 = []
y_coord_0 = []

for i in range(num_preds):
    if (y_test[i] == 1):
        x_coord_1.append(x_test[i][0])
        y_coord_1.append(x_test[i][1])
    else:
        x_coord_0.append(x_test[i][0])
        y_coord_0.append(x_test[i][1])


x_coord = [x[0] for x in x_test]
y_coord = [x[1] for x in x_test]

plt.scatter(x_coord_1, y_coord_1, color = 'red')
plt.scatter(x_coord_0, y_coord_0, color = 'blue')

b = classifier.intercept_[0]
w1, w2 = classifier.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = -1.5, 1.5
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, lw=1, ls='--')

print("Classifier's classification error for the test data is: ", 1 - classifier.score(x_test, y_test))
print("Classifier's classification accuracy for the test data is: ", classifier.score(x_test, y_test))
print("Classifier's classification accuracy for the train data is: ", classifier.score(x_train, y_train))

plt.show()

