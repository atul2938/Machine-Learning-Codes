import numpy as np
import matplotlib.pyplot as mtp
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def find_cost(x, y, theta):
    m=len(x)
    matrix=np.square(np.subtract(np.dot(theta.T, x.T), y.T))
    cost=np.sqrt(np.sum(matrix)/(2*m))
    return cost


def gradient_descent(x, y, x_t, y_t, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    cost_t=[]
    cost_v=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)

        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))

        cost_t.append(find_cost(x, y, theta))
        cost_v.append(find_cost(x_t, y_t, theta))

    return cost_t, cost_v


def normal_equation(x, y, x_t, y_t):
    op1=np.dot(x.T, x)
    op2=np.linalg.inv(op1)
    op3=np.dot(op2, x.T)
    theta= np.dot(op3, y)
    cost=find_cost(x, y, theta)
    cost_v=find_cost(x_t, y_t, theta)
    return cost, cost_v


gender = np.loadtxt('Dataset.data', dtype='str', delimiter=" ", usecols=[0])
table = np.loadtxt('Dataset.data', delimiter=" ", usecols=[1, 2, 3, 4, 5, 6, 7])
label=np.loadtxt('Dataset.data', delimiter=" ", usecols=[8]).reshape(len(gender),1)
gender_num=np.zeros(len(gender))

for t in range(len(gender)):
    if gender[t]=='M':
        gender_num[t]=1
    elif gender[t]=='F':
        gender_num[t]=2
    else:
        gender_num[t]=3

gender_num_col=gender_num.reshape(len(gender), 1)
feature=np.append(gender_num_col, table, axis=1)
notation=np.ones(len(gender)).reshape(len(gender), 1)
feature=np.append(notation, feature, axis=1)

# Feature Normalization
for i in range(1, 9):
    mean=feature[:, [i]].mean()
    std=feature[:, [i]].std()
    for j in range(len(feature)):
        feature[j][i]=(feature[j][i]-mean)/std

partition=len(feature)//5

# 5-fold validation
ne_trainingSet=[]
ne_validationSet=[]

gd_cost_t=[]
gd_cost_v=[]
for i in range(5):
    x_test=feature[i*partition: (i+1)*partition, :]
    y_test=label[i*partition: (i+1)*partition, :]
    x_train=np.delete(feature, slice(i*partition, (i+1)*partition), 0)
    y_train=np.delete(label, slice(i*partition, (i+1)*partition), 0)

    cost_train, cost_validation=gradient_descent(x_train, y_train, x_test, y_test, 0.1, 10000)
    gd_cost_t.append(cost_train)
    gd_cost_v.append(cost_validation)

    cost_t_ne, cost_v_ne=normal_equation(x_train, y_train, x_test, y_test)
    ne_trainingSet.append(cost_t_ne)
    ne_validationSet.append(cost_v_ne)

mean_t=[]
mean_v=[]
iters=[]
for i in range(10000):
    sum_t=0
    sum_v=0
    for j in range(5):
        sum_t+=gd_cost_t[j][i]
        sum_v+=gd_cost_v[j][i]
    mean_t.append(sum_t/5)
    mean_v.append(sum_v/5)
    iters.append(i+1)

mtp.plot(iters, mean_t)
mtp.xlabel("Iterations")
mtp.ylabel("Mean RMSE")
mtp.title("Mean RMSE vs. Gradient Descent Iterations : Training Set")
mtp.show()

mtp.plot(iters, mean_v, color='green')
mtp.xlabel("Iterations")
mtp.ylabel("Mean RMSE")
mtp.title("Mean RMSE vs. Gradient Descent Iterations : Validation Set")
mtp.show()

print("Linear Regression")
print("Normal Equation RMSE:")
for i in range(5):
    print("Training Set: ", ne_trainingSet[i], ", Validation Set: ", ne_validationSet[i])

print("\nFinal RMSE from Gradient Descent: ")
print("Training Set: ", mean_t[9999])
print("Validation Set: ", mean_v[9999])

print("\nFinal RMSE from Normal Equation: ")
print("Training Set: ", sum(ne_trainingSet)/5)
print("Validation Set: ", sum(ne_validationSet)/5)

########################################################################


def ridge_cost(x, y, theta, op_para):
    m=len(x)
    matrix=np.square(np.subtract(np.dot(theta.T, x.T), y.T))
    cost1=np.sum(matrix)
    cost2=np.multiply(np.sum(np.square(theta)), op_para)
    # print("Cost1: ",cost1, "cost 2:", cost2)
    cost=np.sqrt((cost1+ cost2)/(2*m))
    return cost


def ridge_gd(x, y, x_t, y_t, op_para, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    # print("theta shape : ", theta.shape)
    cost_t=[]
    cost_v=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)
        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        # print("Temp Theta : ", temp_theta)
        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))
        for j in range(1, col):
            reg_term=learning_rate*op_para*theta[j][0]/m
            # print("Reg Term", reg_term)
            theta[j][0]=theta[j][0]-reg_term

        cost_t.append(ridge_cost(x, y, theta, op_para))
        cost_v.append(ridge_cost(x_t, y_t, theta, op_para))
        # if(epoch>1):
        #     if(cost[epoch-1]-cost[epoch]<0.001):
        #         print("Epoch Saturation Ridge: ", epoch)
        #         return theta, cost

        # if epoch%100==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)

    return cost_t, cost_v


def lasso_cost(x, y, theta, op_para):
    m=len(x)
    matrix=np.square(np.subtract(np.dot(theta.T, x.T), y.T))
    cost1=np.sum(matrix)
    cost2=np.multiply(np.sum(np.absolute(theta)), op_para)
    # print("Cost1: ",cost1, "cost 2:", cost2)
    cost=np.sqrt((cost1+ cost2)/(2*m))
    return cost


def lasso_gd(x, y, x_t, y_t, op_para, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    # print("theta shape : ", theta.shape)
    cost_t=[]
    cost_v=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.ones(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)
        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        # print("Temp Theta : ", temp_theta)

        for j in range(1, col):
            sign=1
            if theta[j][0]>0:
                sign=1
            elif theta[j][0]<0:
                sign=-1
            else:
                sign=0
            reg_term=learning_rate*op_para*sign/(2*m)
            # print("Reg Term", reg_term)
            theta[j][0]=theta[j][0]-reg_term

        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))

        cost_t.append(lasso_cost(x, y, theta, op_para))
        cost_v.append(lasso_cost(x_t, y_t, theta, op_para))
        # print("Cost v : ", cost_v[epoch])
        # if(epoch>1):
        #     if(cost[epoch-1]-cost[epoch]<0.001):
        #         print("Epoch Saturation Lasso: ", epoch)
        #         return theta, cost
        # if epoch%100==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)

    return cost_t, cost_v


def ridge_gdl(x, y, op_para, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    # print("theta shape : ", theta.shape)
    cost=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)
        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        # print("Temp Theta : ", temp_theta)
        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))
        for j in range(1, col):
            reg_term=learning_rate*op_para*theta[j][0]/m
            # print("Reg Term", reg_term)
            theta[j][0]=theta[j][0]-reg_term

        cost.append(ridge_cost(x, y, theta, op_para))

        # if(epoch>1):
        #     if(cost[epoch-1]-cost[epoch]<0.001):
        #         print("Epoch Saturation Ridge: ", epoch)
        #         return theta, cost

        # if epoch%100==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)

    return theta, cost


def lasso_gdl(x, y, op_para, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    # print("theta shape : ", theta.shape)
    cost=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.ones(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)
        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        # print("Temp Theta : ", temp_theta)

        for j in range(1, col):
            sign=1
            if theta[j][0]>0:
                sign=1
            elif theta[j][0]<0:
                sign=-1
            else:
                sign=0
            reg_term=learning_rate*op_para*sign/(2*m)
            # print("Reg Term", reg_term)
            theta[j][0]=theta[j][0]-reg_term

        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))

        cost.append(lasso_cost(x, y, theta, op_para))

        # if(epoch>1):
        #     if(cost[epoch-1]-cost[epoch]<0.001):
        #         print("Epoch Saturation Lasso: ", epoch)
        #         return theta, cost
        # if epoch%100==0:
        #     print("Epoch ", epoch, "- Cost : ", cost)

    return theta, cost


print("\nRegularization")
print("Validation Set 2 has the lowest RMSE. Holding it out.")

x_test_set=feature[partition: 2*partition, :]
y_test_set=label[partition: 2*partition, :]

feature_new=np.delete(feature, slice(partition, 2*partition), 0)
label_new=np.delete(label, slice(partition, 2*partition), 0)

ridge=Ridge()
lasso=Lasso()

parameter={'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.005, 0.001, 5, 10, 20]}

ridge_reg=GridSearchCV(ridge, parameter, cv=5)
ridge_reg.fit(feature_new, label_new)
# print(ridge_reg.best_params_)
# print(ridge_reg.best_score_)

# print(feature_new.shape, feature_new[:, 1:].shape)
lasso_reg=GridSearchCV(lasso, parameter, cv=5)
lasso_reg.fit(feature_new, label_new)
# print(lasso_reg.best_params_)
# print(lasso_reg.best_score_)

l1_lambda=lasso_reg.best_params_.get("alpha")
l2_lambda=ridge_reg.best_params_.get("alpha")

print("Optimal Parameter for ridge: ", l2_lambda)
print("Optimal Parameter for Lasso: ", l1_lambda)

# theta_ridge, cost_ridge=ridge_gd(feature_new, label_new, 1)
# print("Theta ridge:", theta_ridge, "\nCost ridge: ", cost_ridge)
#
# theta_lasso, cost_lasso=lasso_gd(feature_new, label_new, l1_lambda)
# print("Theta lasso:", theta_lasso, "\nCost lasso: ", cost_lasso)

partition=len(feature_new)//5

gdr_cost_t=[]
gdr_cost_v=[]
gdl_cost_t=[]
gdl_cost_v=[]

for i in range(5):
    x_test=feature_new[i*partition: (i+1)*partition, :]
    y_test=label_new[i*partition: (i+1)*partition, :]
    x_train=np.delete(feature_new, slice(i*partition, (i+1)*partition), 0)
    y_train=np.delete(label_new, slice(i*partition, (i+1)*partition), 0)

    costR_train, costR_validation=ridge_gd(x_train, y_train, x_test, y_test, l2_lambda, 0.2, 1000)
    costL_train, costL_validation=lasso_gd(x_train, y_train, x_test, y_test, l1_lambda, 0.2, 1000)
    gdr_cost_t.append(costR_train)
    gdr_cost_v.append(costR_validation)
    gdl_cost_t.append(costL_train)
    gdl_cost_v.append(costL_validation)

meanr_t=[]
meanr_v=[]

meanl_t=[]
meanl_v=[]

iters=[]
for i in range(1000):
    sumr_t=0
    sumr_v=0
    suml_t=0
    suml_v=0
    for j in range(5):
        sumr_t+=gdr_cost_t[j][i]
        sumr_v+=gdr_cost_v[j][i]
        suml_t+=gdl_cost_t[j][i]
        suml_v+=gdl_cost_v[j][i]
        # print("For sum: ", gdl_cost_v[j][i])
    meanr_t.append(sumr_t/5)
    meanr_v.append(sumr_v/5)
    meanl_t.append(suml_t/5)
    meanl_v.append(suml_v/5)
    # print(meanl_v[i])
    # print('Sum l:', suml_v/5)
    iters.append(i+1)


iters=[]
for i in range(1000):
    iters.append(i+1)

# print("Last Ridge Cost: ", gdr_cost_t[999], "Last lasso cost: ", gdl_cost_t[999])
mtp.plot(iters, meanr_t, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Training Set')
mtp.plot(iters, meanr_v, color='orange')
orange_patch = mpatches.Patch(color='orange', label='Validation Set')
mtp.legend(handles=[blue_patch, orange_patch])
mtp.xlabel("Iterations")
mtp.ylabel("Mean RMSE")
mtp.title("RMSE vs. Iterations : Ridge L2 Regularization")
mtp.show()

mtp.plot(iters, meanl_t, color='blue')
blue_patch = mpatches.Patch(color='blue', label='Training Set')
mtp.plot(iters, meanl_v, color='orange')
orange_patch = mpatches.Patch(color='orange', label='Validation Set')
mtp.legend(handles=[blue_patch, orange_patch])
mtp.xlabel("Iterations")
mtp.ylabel("Mean RMSE")
mtp.title("RMSE vs. Iterations : Lasso L1 Regularization")
mtp.show()


thetaR, gdr_cost_t=ridge_gdl(feature_new, label_new, l2_lambda, 0.2, 1000)
thetaL, gdl_cost_t=lasso_gdl(feature_new, label_new, l1_lambda, 0.2, 1000)
costR_test_set=ridge_cost(x_test_set, y_test_set, thetaR, l2_lambda)
costL_test_set=lasso_cost(x_test_set, y_test_set, thetaL, l1_lambda)

print("Cost with L2 Regularization on Test Set: ", costR_test_set)
print("Cost with L1 Regularization on Test Set: ", costL_test_set)

##############################################################################


def gradient_descent(x, y, learning_rate=0.1, iterations=1000):
    col=x.shape[1]
    theta=np.zeros(col).reshape(col, 1)
    cost_t=[]
    m=len(x)
    for epoch in range(iterations):
        temp_theta=np.zeros(col).reshape(col, 1)
        prediction=np.dot(theta.T, x.T)
        diff=np.subtract(prediction, y.T)

        for j in range(col):
            j_theta=np.dot(diff, x[:, [j]])
            temp_theta[j][0]=j_theta

        theta=np.subtract(theta, np.multiply(np.true_divide(temp_theta, m), learning_rate))
        cost_t.append(find_cost(x, y, theta))

    return cost_t, theta


print("\nBest Fit Line: ")

brain=np.loadtxt('data.csv', delimiter=",", skiprows=1, usecols=[0]).reshape(167, 1)
body=np.loadtxt('data.csv', delimiter=",", skiprows=1, usecols=[1]).reshape(167, 1)
# print(brain.shape, body.shape)

b0=np.ones(len(brain)).reshape(len(brain), 1)
brain_x=np.append(b0, brain, axis=1)

# print("Brain x: ", brain_x.shape)
# print("Body: ", body.shape)
cost_bfl, theta_bfl=gradient_descent(brain_x, body, 0.00001, 1000)
# print("Cost of best fit line (without regularisation): ", cost_bfl)

x_axis=brain.flatten()
y_axis=np.dot(theta_bfl.T, brain_x.T).flatten()
# print("x_axis: ", x_axis, "y_axis: ", y_axis)
# print("x_shape: ", x_axis.shape, "y_shape: ", y_axis.shape)
# print("x_list: ", len(list(x_axis)), "y_list: ", len(list(y_axis)))
mtp.scatter(brain, body, color='skyblue')
mtp.xlabel("brain-weight")
mtp.ylabel("body-weight")
mtp.title("Best Fit Line")

mtp.plot(x_axis, y_axis, color='green')
green_patch = mpatches.Patch(color='green', label='Without Regularization')

# mtp.show()
parameter={'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.2, 0.3, 0.1, 0.005, 0.001, 5, 10, 20]}

ridge_reg_bfl=GridSearchCV(ridge, parameter, cv=5)
ridge_reg_bfl.fit(brain_x, body)

lasso_reg=GridSearchCV(lasso, parameter, cv=5)
lasso_reg.fit(brain_x, body)

l1_lambda_bfl=lasso_reg.best_params_.get("alpha")
l2_lambda_bfl=ridge_reg.best_params_.get("alpha")

print("Optimal Parameter for ridge: ", l2_lambda_bfl)
print("Optimal Parameter for Lasso: ", l1_lambda_bfl)

thetaR_bfl, gdr_cost_bfl=ridge_gdl(brain_x, body, l2_lambda_bfl, 0.0005, 100000)
thetaL_bfl, gdl_cost_bfl=lasso_gdl(brain_x, body, l1_lambda_bfl, 0.0005, 100000)
# print("thetaR_bfl: ", thetaR_bfl)
yR_axis=np.dot(thetaR_bfl.T, brain_x.T).flatten()
yL_axis=np.dot(thetaL_bfl.T, brain_x.T).flatten()
# print("x-axis: ",x_axis.shape )
# print("yR_axis: ", yR_axis.shape)
# print("yR_axis: ", yR_axis.shape)
print("Cost without Regularization: ", cost_bfl[999])
print("Cost with L2 Regularization: ", gdr_cost_bfl[99999])
print("Cost with L1 Regularization: ", gdl_cost_bfl[99999])

mtp.plot(x_axis, yR_axis, color="black")
black_patch = mpatches.Patch(color='black', label='L2 Regularization')

mtp.plot(x_axis, yL_axis, color="yellow")
yellow_patch = mpatches.Patch(color='yellow', label='L1 Regularization')

mtp.legend(handles=[green_patch, black_patch, yellow_patch])
mtp.show()
