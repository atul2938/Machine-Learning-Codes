from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

train = pd.read_csv('mnist_train.csv', nrows=30000)
test = pd.read_csv('mnist_test.csv')

# print(train)
# print(test)

feature_train=train.iloc[:, 1:785]
label_train=train.iloc[:, 0]

feature_train=feature_train.values
label_train=label_train.values.reshape(len(feature_train), 1)
# print(feature_train, label_train.shape)
feature_test=test.iloc[:, 1:785].values
label_test=test.iloc[:, 0].values.reshape(len(feature_test), 1)
# print(feature_train.shape)

y_train=[]
for i in range(10):
    m=[]
    for j in range(len(label_train)):
        if label_train[j][0]==i:
            m.append(0)
        else:
            m.append(1)
    y_train.append(m)
y_train=np.asarray(y_train)
y_train=y_train.T

y_test=[]
for i in range(10):
    m=[]
    for j in range(len(label_test)):
        if label_test[j][0]==i:
            m.append(0)
        else:
            m.append(1)
    y_test.append(m)
y_test=np.asarray(y_test)
y_test=y_test.T


lr_lasso=LogisticRegression(penalty='l1')
ovr_lasso=OneVsRestClassifier(estimator=lr_lasso)
ans_l1_train=[]
ans_l1_test=[]
for i in range(10):
    res=ovr_lasso.fit(feature_train, y_train[:, [i]])
    sc_train=res.score(feature_train, y_train[:, [i]])
    sc_test=res.score(feature_test, y_test[:, [i]])
    ans_l1_train.append(sc_train*100)
    ans_l1_test.append(sc_test*100)

print("L1 Regularization")
for i in range(10):
    print("Training Score for ", i, ": ", ans_l1_train[i])
    print("Testing Score for ", i, ": ", ans_l1_test[i])

lr_ridge=LogisticRegression(solver='lbfgs', penalty='l2')
ovr_ridge=OneVsRestClassifier(estimator=lr_ridge)
ans_l2_train=[]
ans_l2_test=[]

for i in range(10):
    res=ovr_ridge.fit(feature_train, y_train[:, [i]])
    sc_train=res.score(feature_train, y_train[:, [i]])
    sc_test=res.score(feature_test, y_test[:, [i]])
    ans_l2_train.append(sc_train*100)
    ans_l2_test.append(sc_test*100)

print("\nL2 Regularization")
for i in range(10):
    print("Training Score for ", i, ": ", ans_l2_train[i])
    print("Testing Score for ", i, ": ", ans_l2_test[i])

