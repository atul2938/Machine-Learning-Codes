import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction import DictVectorizer

import warnings
warnings.simplefilter("ignore")

cols=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
df1=pd.read_csv('train.csv', names=cols)
# print(df.head())
# print(df.dtypes)
df2=pd.read_csv('test.csv', names=cols)
# print(df2.head)
df=pd.concat([df1, df2])
# print(df)

feature=df.iloc[:, 0:14]
label_tv=df1.iloc[:, 14]
label_test=df2.iloc[:, 14]
# print("feature: ", feature.shape)
# print("label: ", label_train, label_test)
# obj_df = df.select_dtypes(include=['object']).copy()
# print(obj_df.head())
# print(obj_df.dtypes)
# lb_make = LabelEncoder()
# obj_df["make_code"] = lb_make.fit_transform(obj_df["make"])
# 3, 5, 6, 7, 8, 9, 13, 14]
# ohe=OneHotEncoder(categorical_features=[1])
# A=ohe.fit_transform(df).toarray()
# print(A)

# dv_X = DictVectorizer(sparse=False)
# feature_dict=feature.to_dict(orient='records')
# feature_encoded = dv_X.fit_transform(feature_dict)
# print(feature_encoded.shape)
# print(df_encoded)

le = LabelEncoder()

feature_encoded=feature.apply(le.fit_transform)
# print(feature_encoded)

label_tv=le.fit_transform(label_tv)
label_test=le.fit_transform(label_test)
# print("Label Encoded: ", label_encoded.shape)
# print(label_train)
# print(label_test)

feature_encoded=feature_encoded.values
# feature_encoded=feature_encoded[0:5]

# print("feature encoded: ", feature_encoded)

# label_encoded=label_encoded[0:5]

# print("label encoded: ", label_encoded)

num=len(feature_encoded)
print("Num: ", num)
x0=np.ones(num).reshape(num, 1)
feature_encoded=np.append(x0, feature_encoded, axis=1)
label_tv=label_tv.reshape(len(label_tv), 1)
label_test=label_test.reshape(len(label_test), 1)
print("Feature Encoded Shape: ", feature_encoded.shape)
print("label tv Shape: ", label_tv.shape)
print("label test Shape: ", label_test.shape)


def cost_log(x, y, theta):
    m=len(x)
    op1=hypothesis(theta, x)
    cost1=np.multiply(y, np.log(op1).T)
    cost2=np.multiply(1-y, np.log(1-op1).T)
    cost=-np.true_divide(np.sum(cost1)+np.sum(cost2), m)
    return cost


def hypothesis(theta, x):
    op1=np.dot(theta.T, x.T)
    h=np.true_divide(1, 1+np.exp(-op1))
    return h


def log_reg(x, y, learning_rate=0.0001, iterations=1000):
    col=x.shape[1]
    m=len(x)
    theta=np.ones(col).reshape(col, 1)
    cost=0.0
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=hypothesis(theta, x)
        diff=np.subtract(prediction, y.T)
        temp_theta=np.dot(diff, x).T
        theta=theta-np.multiply(temp_theta/m, learning_rate)
        cost=cost_log(x, y, theta)

        # if epoch%500==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)
            # print(theta)

    return theta, cost


def accuracy(theta, x, y, th):
    prediction=hypothesis(theta, x).flatten()
    # print(prediction.shape)
    y_actual=y.flatten()

    count=0
    for m in range(len(y_actual)):
        p=0
        # print(prediction[m])
        if prediction[m]>th:
            p=1
        if p==y_actual[m]:
            count+=1
        # if y_actual[m]!=0 | y_actual[m]!=1:
        #     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

    return (count*100)/len(y_actual)


# Feature Normalization


# print("Final Feature encoded: ", feature_encoded[0])


for i in range(1, 15):
    mean=feature_encoded[:, [i]].mean()
    std=feature_encoded[:, [i]].std()
    for j in range(len(feature_encoded)):
        feature_encoded[j][i]=(feature_encoded[j][i]-mean)/std


feature_train=feature_encoded[0:24130, :]
feature_validation=feature_encoded[24130:, :]
feature_test=feature_encoded[30162:, :]

label_train=label_tv[0:24130, :]
label_validation=label_tv[24130:, : ]
# label_test=label_encoded[30162:, :]

print("\nTrain Shapes:", feature_train.shape, label_train.shape)
print("Test Shapes:", feature_test.shape, label_test.shape)

thetaLog, cost_Log=log_reg(feature_train, label_train, 0.001, 3000)

acc=accuracy(thetaLog, feature_train, label_train, 0.90)
print("\nAccuracy without regularization on training set: ", acc)

acc=accuracy(thetaLog, feature_validation, label_validation, 0.90)
print("Accuracy without regularization on validation set: ", acc)

acc=accuracy(thetaLog, feature_test, label_test, 0.90)
print("Accuracy without regularization on testing set: ", acc)

# acc=accuracy(thetaLog, feature_train, label_train, 0.80)
# print("Accuracy: ", acc)
#
# acc=accuracy(thetaLog, feature_train, label_train, 0.70)
# print("Accuracy: ", acc)
#
# acc=accuracy(thetaLog, feature_train, label_train, 0.60)
# print("Accuracy: ", acc)
#
# acc=accuracy(thetaLog, feature_train, label_train, 0.50)
# print("Accuracy: ", acc)
#
# acc=accuracy(thetaLog, feature_train, label_train, 0.40)
# print("Accuracy: ", acc)
#
# acc=accuracy(thetaLog, feature_train, label_train, 0.30)
# print("Accuracy: ", acc)

####################################################################


def cost_ridge(x, y, theta, op_para):
    m=len(x)
    op1=hypothesis(theta, x)
    cost1=np.multiply(y, np.log(op1).T)
    cost2=np.multiply(1-y, np.log(1-op1).T)
    cost3=np.true_divide(np.multiply(np.sum(np.square(theta)), op_para), 2*m)
    cost=-np.true_divide(np.sum(cost1)+np.sum(cost2), m)
    cost+=cost3
    return cost


def log_ridge(x, y, x_t, y_t, op_para, learning_rate=0.0001, iterations=1000):
    col=x.shape[1]
    m=len(x)
    theta=np.ones(col).reshape(col, 1)
    cost=[]
    accu=[]
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=hypothesis(theta, x)
        diff=np.subtract(prediction, y.T)
        temp_theta=np.dot(diff, x).T
        theta=theta-np.multiply(np.subtract(temp_theta, theta*op_para)/m, learning_rate)

        cost.append(cost_ridge(x_t, y_t, theta, op_para))
        accu.append(accuracy(theta, x_t, y_t, 0.9))
        # if epoch%1000==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)
            # print(theta)

    return theta, cost, accu


ridge=Ridge()
lasso=Lasso()

parameter={'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.005, 0.001, 5, 10, 20]}

ridge_reg=GridSearchCV(ridge, parameter, cv=5)
ridge_reg.fit(feature_train, label_train)

lasso_reg=GridSearchCV(lasso, parameter, cv=5)
lasso_reg.fit(feature_train, label_train)

l1_lambda=lasso_reg.best_params_.get("alpha")
l2_lambda=ridge_reg.best_params_.get("alpha")

# print("\nOptimal Parameter for ridge: ", l2_lambda)
# print("Optimal Parameter for Lasso: ", l1_lambda)

thetaLogR, cost_LogR, acc_R=log_ridge(feature_train, label_train, feature_test, label_test, l2_lambda,  0.01, 1000)

acc=accuracy(thetaLogR, feature_train, label_train, 0.90)
print("\nAccuracy with L2 Regularization on Training Set: ", acc)

acc=accuracy(thetaLogR, feature_validation, label_validation, 0.90)
print("Accuracy with L2 Regularization on Validation Set: ", acc)

acc=accuracy(thetaLogR, feature_test, label_test, 0.90)
print("Accuracy with L2 Regularization on Testing Set: ", acc)


########################################################################


def cost_lasso(x, y, theta, op_para):
    m=len(x)
    op1=hypothesis(theta, x)
    cost1=np.multiply(y, np.log(op1).T)
    cost2=np.multiply(1-y, np.log(1-op1).T)
    cost3=np.true_divide(np.multiply(np.sum(np.absolute(theta)), op_para), 2*m)
    cost=-np.true_divide(np.sum(cost1)+np.sum(cost2), m)
    cost+=cost3
    return cost


def log_lasso(x, y, x_t, y_t, op_para, learning_rate=0.0001, iterations=1000):
    col=x.shape[1]
    m=len(x)
    theta=np.ones(col).reshape(col, 1)
    cost=[]
    acc=[]
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=hypothesis(theta, x)
        diff=np.subtract(prediction, y.T)
        temp_theta=np.dot(diff, x).T

        reg_term=np.zeros(col).reshape(col, 1)
        for j in range(1, col):
            sign=1
            if theta[j][0]>0:
                sign=1
            elif theta[j][0]<0:
                sign=-1
            else:
                sign=0
            reg_term[j][0]=op_para*sign/2

        theta=theta-np.multiply(np.add(np.subtract(temp_theta, theta*op_para), reg_term)/m, learning_rate)
        cost.append(cost_lasso(x_t, y_t, theta, op_para))
        acc.append(accuracy(theta, x_t, y_t, 0.9))
        # if epoch%1000==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)
            # print(theta)

    return theta, cost, acc


thetaLogL, cost_LogL, acc_L=log_ridge(feature_train, label_train, feature_test, label_test, l1_lambda, 0.01, 1000)

acc=accuracy(thetaLogL, feature_train, label_train, 0.90)
print("\nAccuracy with L1 Regularization on Training Set: ", acc)

acc=accuracy(thetaLogL, feature_validation, label_validation, 0.90)
print("Accuracy with L1 Regularization on Validation Set: ", acc)

acc=accuracy(thetaLogL, feature_test, label_test, 0.90)
print("Accuracy with L1 Regularization on Testing Set: ", acc)

iters=[]
for i in range(1000):
    iters.append(i+1)

mtp.plot(iters, cost_LogR, color='green')
green_patch = mpatches.Patch(color='green', label='L2 Regularization')
mtp.plot(iters, cost_LogL, color='blue')
blue_patch = mpatches.Patch(color='blue', label='L1 Regularization')
mtp.legend(handles=[green_patch, blue_patch])
mtp.xlabel("error")
mtp.ylabel("iterations")
mtp.title("Logistic Regression : Error vs Iterations (Testing Set)")
mtp.show()

mtp.plot(acc_R, iters, color='green')
green_patch = mpatches.Patch(color='green', label='L2 Regularization')
mtp.plot(acc_L, iters, color='blue')
blue_patch = mpatches.Patch(color='blue', label='L1 Regularization')
mtp.legend(handles=[green_patch, blue_patch])
mtp.xlabel("accuracy (%)")
mtp.ylabel("iterations")
mtp.title("Logistic Regression : Accuracy vs Iterations (Testing Set)")
mtp.show()
