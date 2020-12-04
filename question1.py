import numpy as np
import h5py as h5
import matplotlib.pyplot as mtp
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import matplotlib.patches as mpatches

import warnings
warnings.simplefilter("ignore")


# Referred from https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Refered from https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def rbf(x, y, sigma=1.0):
    return np.exp(- np.sum(np.power((x - y), 2)) / 2*(sigma**2))


def rbf_kernel(x, y, sigma=1.0):
    # print("X shape: ", x.shape, "Y shape: ", y.shape)
    kernel_gram_matrix = np.zeros(shape=(len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel_gram_matrix[i][j] = rbf(x[i], y[j], sigma)
    return kernel_gram_matrix


# Data 1
data1 = h5.File('data_1.h5', 'r')
d1x = data1.get("x")[()]
d1y = data1.get("y")[()]
# print("Data 1 shape: ", d1y.shape)

# d1x0 = np.zeros(shape=(np.count_nonzero(d1y == 0), 2))
# d1x1 = np.zeros(shape=(np.count_nonzero(d1y == 1), 2))
# d1y0 = np.zeros(np.count_nonzero(d1y == 0))
# d1y1 = np.ones(np.count_nonzero(d1y == 1))
#
# a = 0
# b = 0
# for i in range(100):
#     if d1y[i] == 0:
#         d1x0[a][0], d1x0[a][1] = d1x[i][0], d1x[i][1]
#         a = a+1
#     else:
#         d1x1[b][0], d1x1[b][1] = d1x[i][0], d1x[i][1]
#         b = b+1

ax = mtp.axes()
ax.scatter(d1x[:, 0], d1x[:, 1], c=d1y, edgecolors='black')
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 1 plot")
mtp.show()

model = svm.SVC(kernel=rbf_kernel, C=1.0)
model.fit(d1x, d1y)
predicted = model.predict(d1x)
print("Accuracy for data 1: ", accuracy_score(d1y, predicted))

# fig=mtp.figure()
ax = mtp.axes()
xx, yy = make_meshgrid(d1x[:, 0], d1x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(d1x[:, 0], d1x[:, 1], c=d1y, edgecolors='black')
# plot_decision_regions(d1x, d1y, clf=model, legend=2)
# mtp.scatter(d1x0[:, [0]], d1x0[:, [1]], color="red")
# mtp.scatter(d1x1[:, [0]], d1x1[:, [1]], color="blue")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 1 plot with decision boundary (rbf kernel)")
mtp.show()


# Data 2
data2 = h5.File('data_2.h5', 'r')
d2x = data2.get("x")[()]
d2y = data2.get("y")[()]
# print("Data 2 shape: ", d2y.shape)

# d2x0 = np.zeros(shape=(np.count_nonzero(d2y == 0), 2))
# d2x1 = np.zeros(shape=(np.count_nonzero(d2y == 1), 2))
# d2y0 = np.zeros(np.count_nonzero(d2y == 0))
# d2y1 = np.ones(np.count_nonzero(d2y == 1))
#
# a = 0
# b = 0
# for i in range(100):
#     if d2y[i] == 0:
#         d2x0[a][0], d2x0[a][1] = d2x[i][0], d2x[i][1]
#         a = a+1
#     else:
#         d2x1[b][0], d2x1[b][1] = d2x[i][0], d2x[i][1]
#         b = b+1


ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.scatter(d2x[:, 0], d2x[:, 1], c=d2y, edgecolors='black')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 2 plot")
mtp.show()

model = svm.SVC(kernel=rbf_kernel, C=10.0, gamma='auto')
model.fit(d2x, d2y)
predicted = model.predict(d2x)
print("Accuracy for data 2: ", accuracy_score(d2y, predicted))

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
xx, yy = make_meshgrid(d2x[:, 0], d2x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(d2x[:, 0], d2x[:, 1], c=d2y, edgecolors='black')
# ax.set_zlabel('y')
# mtp.scatter(d2x0[:, [0]], d2x0[:, [1]], color="red")
# mtp.scatter(d2x1[:, [0]], d2x1[:, [1]], color="blue")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 2 plot with decision boundary (rbf kernel)")
mtp.show()


# Data 3
data3 = h5.File('data_3.h5', 'r')
d3x = data3.get("x")[()]
d3y = data3.get("y")[()]
# print("Data 3 shape: ", d3y.shape)

# d3x0 = np.zeros(shape=(np.count_nonzero(d3y == 0), 2))
# d3x1 = np.zeros(shape=(np.count_nonzero(d3y == 1), 2))
# d3x2 = np.zeros(shape=(np.count_nonzero(d3y == 2), 2))
# d3y0 = np.zeros(np.count_nonzero(d3y == 0))
# d3y1 = np.ones(np.count_nonzero(d3y == 1))
# d3y2 = np.empty(np.count_nonzero(d3y == 2))
# d3y2.fill(2)
#
# a = 0
# b = 0
# c = 0
# for i in range(100):
#     if d3y[i] == 0:
#         d3x0[a][0], d3x0[a][1] = d3x[i][0], d3x[i][1]
#         a = a+1
#     elif d3y[i] == 1:
#         d3x1[b][0], d3x1[b][1] = d3x[i][0], d3x[i][1]
#         b = b+1
#     else:
#         d3x2[b][0], d3x2[b][1] = d3x[i][0], d3x[i][1]
#         c = c+1

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.scatter(d3x[:, 0], d3x[:, 1], c=d3y, edgecolors='black')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
blue_patch = mpatches.Patch(color='skyblue', label='class 2')
mtp.legend(handles=[purple_patch, yellow_patch, blue_patch])
mtp.title("Data 3 plot")
mtp.show()

model = svm.SVC(kernel='linear', C=1.0, gamma='auto')
model.fit(d3x, d3y)
predicted = model.predict(d3x)
print("Accuracy for data 3: ", accuracy_score(d3y, predicted))

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
xx, yy = make_meshgrid(d3x[:, 0], d3x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(d3x[:, 0], d3x[:, 1], c=d3y, edgecolors='black')
# ax.set_zlabel('y')
# mtp.scatter(d3x0[:, [0]], d3x0[:, [1]], color="red")
# mtp.scatter(d3x1[:, [0]], d3x1[:, [1]], color="blue")
# mtp.scatter(d3x2[:, [0]], d3x2[:, [1]], color="green")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
blue_patch = mpatches.Patch(color='skyblue', label='class 2')
mtp.legend(handles=[purple_patch, yellow_patch, blue_patch])
mtp.title("Data 3 plot with decision boundary (linear kernel)")
mtp.show()


# Data 4
data4 = h5.File('data_4.h5', 'r')
d4x = data4.get("x")[()]
d4y = data4.get("y")[()]
# print("Data 4 shape: ", d4y.shape)

d4x0 = np.zeros(shape=(np.count_nonzero(d4y == 0), 2))
d4x1 = np.zeros(shape=(np.count_nonzero(d4y == 1), 2))
d4y0 = np.zeros(np.count_nonzero(d4y == 0))
d4y1 = np.ones(np.count_nonzero(d4y == 1))

a = 0
b = 0
for i in range(2000):
    if d4y[i] == 0:
        d4x0[a][0], d4x0[a][1] = d4x[i][0], d4x[i][1]
        a = a+1
    else:
        d4x1[b][0], d4x1[b][1] = d4x[i][0], d4x[i][1]
        b = b+1

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.scatter(d4x[:, 0], d4x[:, 1], c=d4y, edgecolors='black')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 4 plot")
mtp.show()

model = svm.SVC(kernel=rbf_kernel, C=1.0, gamma='auto')
model.fit(d4x, d4y)
predicted = model.predict(d4x)
print("Accuracy for data 4: ", accuracy_score(d4y, predicted))

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
xx, yy = make_meshgrid(d4x[:, 0], d4x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(d4x[:, 0], d4x[:, 1], c=d4y, edgecolors='black')
# mtp.scatter(d4x0[:, [0]], d4x0[:, [1]], color="red")
# mtp.scatter(d4x1[:, [0]], d4x1[:, [1]], color="blue")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 4 plot with decision boundary (rbf kernel)")
mtp.show()


# Data5
data5 = h5.File('data_5.h5', 'r')
d5x = data5.get("x")[()]
d5y = data5.get("y")[()]
# print("Data 5 shape: ", d5y.shape)

d5x0 = np.zeros(shape=(np.count_nonzero(d5y == 0), 2))
d5x1 = np.zeros(shape=(np.count_nonzero(d5y == 1), 2))
d5y0 = np.zeros(np.count_nonzero(d5y == 0))
d5y1 = np.ones(np.count_nonzero(d5y == 1))

a = 0
b = 0
for i in range(2000):
    if d5y[i] == 0:
        d5x0[a][0], d5x0[a][1] = d5x[i][0], d5x[i][1]
        a = a+1
    else:
        d5x1[b][0], d5x1[b][1] = d5x[i][0], d5x[i][1]
        b = b+1

# ax = mtp.axes(projection="3d")
# ax.scatter3D(d5x0[:, [0]], d5x0[:, [1]], d5y0, c='r', marker='o')
# ax.scatter3D(d5x1[:, [0]], d5x1[:, [1]], d5y1, c='b', marker='^')

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.scatter(d5x[:, 0], d5x[:, 1], c=d5y, edgecolors='black')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 5 plot")
mtp.show()

model = svm.SVC(kernel=rbf_kernel, C=100.0, gamma='auto')
model.fit(d5x, d5y)
predicted = model.predict(d5x)
print("Accuracy for data 5: ", accuracy_score(d5y, predicted))

ax = mtp.axes()
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
xx, yy = make_meshgrid(d5x[:, 0], d5x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(d5x[:, 0], d5x[:, 1], c=d5y, edgecolors='black')
# mtp.scatter(d5x0[:, [0]], d5x0[:, [1]], color="red")
# mtp.scatter(d5x1[:, [0]], d5x1[:, [1]], color="blue")
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Data 5 plot with decision boundary (rbf_kernel)")
mtp.show()


# Removing outliers from data 4
model = svm.OneClassSVM(kernel=rbf_kernel, nu=0.9, gamma='auto')
out4_0 = model.fit_predict(d4x0)
# print(out4_0)
out4_1 = model.fit_predict(d4x1)

outd4x = np.zeros(shape=(np.count_nonzero(out4_0 == 1) + np.count_nonzero(out4_1 == 1), 2))
outd4y = np.zeros((np.count_nonzero(out4_0 == 1) + np.count_nonzero(out4_1 == 1)))
# d4o = np.zeros(shape=(np.count_nonzero(out4_0 == -1), 2))
# d4y0 = np.zeros(np.count_nonzero(d4y == 0))
# d4y1 = np.ones(np.count_nonzero(d4y == 1))


a = 0
for i in range(len(d4x0)):
    if out4_0[i] == 1:
        outd4x[a][0], outd4x[a][1] = d4x0[i][0], d4x0[i][1]
        outd4y[a] = 0
        a = a+1

for i in range(len(d4x1)):
    if out4_1[i] == 1:
        outd4x[a][0], outd4x[a][1] = d4x1[i][0], d4x1[i][1]
        outd4y[a] = 1
        a = a+1

model = svm.SVC(kernel=rbf_kernel, C=1000.0, gamma='auto')
model.fit(outd4x, outd4y)
predicted = model.predict(outd4x)
print("Accuracy for outlier removed data 4 using One class SVM (rbf kernel): ", accuracy_score(outd4y, predicted))

ax = mtp.axes()
xx, yy = make_meshgrid(outd4x[:, 0], outd4x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(outd4x[:, 0], outd4x[:, 1], c=outd4y, edgecolors='black')
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Outlier removed data 4 plot with decision boundary using One class SVM (rbf kernel)")
mtp.show()


# Removing outliers from data 5
model = svm.OneClassSVM(kernel=rbf_kernel, nu=0.7, gamma='auto')
out5_0 = model.fit_predict(d5x0)
# print(out4_0)
out5_1 = model.fit_predict(d5x1)

outd5x = np.zeros(shape=(np.count_nonzero(out5_0 == 1) + np.count_nonzero(out5_1 == 1), 2))
outd5y = np.zeros((np.count_nonzero(out5_0 == 1) + np.count_nonzero(out5_1 == 1)))

a = 0
for i in range(len(d5x0)):
    if out5_0[i] == 1:
        outd5x[a][0], outd5x[a][1] = d5x0[i][0], d5x0[i][1]
        outd5y[a] = 0
        a = a+1

for i in range(len(d5x1)):
    if out5_1[i] == 1:
        outd5x[a][0], outd5x[a][1] = d5x1[i][0], d5x1[i][1]
        outd5y[a] = 1
        a = a+1

model = svm.SVC(kernel='linear', C=1.0, gamma='auto')
model.fit(outd5x, outd5y)
predicted = model.predict(outd5x)
print("Accuracy for outlier removed data 5 using One class SVM (linear kernel):", accuracy_score(outd5y, predicted))

ax = mtp.axes()
xx, yy = make_meshgrid(outd5x[:, 0], outd5x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(outd5x[:, 0], outd5x[:, 1], c=outd5y, edgecolors='black')
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Outlier removed data 5 plot with decision boundary using One class SVM (linear kernel)")
mtp.show()


# Using Isolation Forest
model = ensemble.IsolationForest(contamination=0.52)
out4_0 = model.fit_predict(d4x0)
# print(out4_0)
out4_1 = model.fit_predict(d4x1)

outd4x = np.zeros(shape=(np.count_nonzero(out4_0 == 1) + np.count_nonzero(out4_1 == 1), 2))
outd4y = np.zeros((np.count_nonzero(out4_0 == 1) + np.count_nonzero(out4_1 == 1)))
# d4o = np.zeros(shape=(np.count_nonzero(out4_0 == -1), 2))
# d4y0 = np.zeros(np.count_nonzero(d4y == 0))
# d4y1 = np.ones(np.count_nonzero(d4y == 1))


a = 0
for i in range(len(d4x0)):
    if out4_0[i] == 1:
        outd4x[a][0], outd4x[a][1] = d4x0[i][0], d4x0[i][1]
        outd4y[a] = 0
        a = a+1

for i in range(len(d4x1)):
    if out4_1[i] == 1:
        outd4x[a][0], outd4x[a][1] = d4x1[i][0], d4x1[i][1]
        outd4y[a] = 1
        a = a+1

model = svm.SVC(kernel=rbf_kernel, C=1000.0, gamma='auto')
model.fit(outd4x, outd4y)
predicted = model.predict(outd4x)
print("Accuracy for outlier removed data 4 using Isolation forest (rbf kernel): ", accuracy_score(outd4y, predicted))

ax = mtp.axes()
xx, yy = make_meshgrid(outd4x[:, 0], outd4x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(outd4x[:, 0], outd4x[:, 1], c=outd4y, edgecolors='black')
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Outlier removed data 4 plot with decision boundary using Isolation forest (rbf kernel)")
mtp.show()


model = ensemble.IsolationForest(contamination=0.64)
out5_0 = model.fit_predict(d5x0)
out5_1 = model.fit_predict(d5x1)

outd5x = np.zeros(shape=(np.count_nonzero(out5_0 == 1) + np.count_nonzero(out5_1 == 1), 2))
outd5y = np.zeros((np.count_nonzero(out5_0 == 1) + np.count_nonzero(out5_1 == 1)))

a = 0
for i in range(len(d5x0)):
    if out5_0[i] == 1:
        outd5x[a][0], outd5x[a][1] = d5x0[i][0], d5x0[i][1]
        outd5y[a] = 0
        a = a+1

for i in range(len(d5x1)):
    if out5_1[i] == 1:
        outd5x[a][0], outd5x[a][1] = d5x1[i][0], d5x1[i][1]
        outd5y[a] = 1
        a = a+1

model = svm.SVC(kernel='linear', C=10.0, gamma='auto')
model.fit(outd5x, outd5y)
predicted = model.predict(outd5x)
print("Accuracy for outlier removed data 5 using Isolation forest (linear kernel): ", accuracy_score(outd5y, predicted))

ax = mtp.axes()
xx, yy = make_meshgrid(outd5x[:, 0], outd5x[:, 1])
plot_contours(ax, model, xx, yy, cmap=mtp.cm.coolwarm)
ax.scatter(outd5x[:, 0], outd5x[:, 1], c=outd5y, edgecolors='black')
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
purple_patch = mpatches.Patch(color='purple', label='class 0')
yellow_patch = mpatches.Patch(color='yellow', label='class 1')
mtp.legend(handles=[purple_patch, yellow_patch])
mtp.title("Outlier removed data 5 plot with decision boundary using Isolation forest (linear kernel)")
mtp.show()


# 80-20 split for data 4
data4_xtrain = d4x[:1600, :]
data4_ytrain = d4y[:1600]
data4_xtest = d4x[1600:, :]
data4_ytest = d4y[1600:]

model = svm.SVC(kernel='linear', gamma='auto')
model.fit(data4_xtrain, data4_ytrain)
# print(model.support_vectors_)
# print(model.n_support_)
# print(model.support_)
# print(model.coef_)
# print(model.intercept_)

predicted = model.predict(data4_xtest)
# print(accuracy_score(data4_ytest, predicted))
correct = 0
for i in range(len(predicted)):
    if predicted[i] == data4_ytest[i]:
        correct += 1

print("For data 4: ")
print("Accuracy by SVM predict function (linear kernel): ", correct/(len(predicted)))

w = model.coef_
b = model.intercept_


def custom_predict_linear(x, w, b):
    if (w[0][0]*x[0] + w[0][1]*x[1] + b) >= 0:
        return 1
    else:
        return 0


correct = 0
for i in range(len(data4_xtest)):
    if data4_ytest[i] == custom_predict_linear(data4_xtest[i], w, b):
        correct += 1

print("Accuracy by implemented predict function (linear kernel): ", correct/(len(data4_xtest)))


model = svm.SVC(kernel='rbf', gamma='auto')
model.fit(data4_xtrain, data4_ytrain)
predicted = model.predict(data4_xtest)
correct = 0
for i in range(len(predicted)):
    if predicted[i] == data4_ytest[i]:
        correct += 1

print("Accuracy by SVM predict function (rbf kernel): ", correct/(len(predicted)))
# print(model.dual_coef_)
# print(model.support_vectors_)
# print(model.intercept_)


# model = svm.SVC(kernel="precomputed", gamma='auto')
# gm = rbf_kernel(data4_xtrain, data4_xtrain)
# model.fit(gm, data4_ytrain)
# predicted = model.predict(rbf_kernel(data4_xtest, data4_xtrain))

model = svm.SVC(kernel=rbf_kernel, gamma="auto")
model.fit(data4_xtrain, data4_ytrain)

y_alpha = model.dual_coef_
k_i = model.support_
rho = model.intercept_
#
# print(model.support_)
# print(model.support_.shape)
# print("y alpha shape: ", y_alpha.shape, "k_i shape: ", k_i.shape, "number: ", model.n_support_)


def custom_predict_rbf(x, y_alpha, k_i, rho, X):
    k = np.zeros(len(k_i))
    # print("Y_alpha shape: ", y_alpha.shape, "k shape: ", k.shape)
    for i in range(len(k_i)):
        k[i] = rbf(X[k_i[i]], x)

    res = np.dot(y_alpha, k) + rho

    if res >= 0:
        return 1
    else:
        return 0


correct = 0
for i in range(len(data4_xtest)):
    if custom_predict_rbf(data4_xtest[i], y_alpha, k_i, rho, data4_xtrain) == data4_ytest[i]:
        correct += 1

print("Accuracy by implemented predict function (implemented rbf kernel): ", correct/(len(predicted)))

# model = svm.SVC(kernel='rbf', gamma="auto")
# model.fit(d1x, d1y)
# print(model.dual_coef_)
# print(model.support_)
# print(model.n_support_)


# 80-20 split for Data 5
data5_xtrain = d5x[:1600, :]
data5_ytrain = d5y[:1600]
data5_xtest = d5x[1600:, :]
data5_ytest = d5y[1600:]

model = svm.SVC(kernel='linear', gamma='auto')
model.fit(data5_xtrain, data5_ytrain)
# print(model.support_vectors_)
# print(model.n_support_)
# print(model.support_)
# print(model.coef_)
# print(model.intercept_)

predicted = model.predict(data5_xtest)
# print(accuracy_score(data4_ytest, predicted))
correct = 0
for i in range(len(predicted)):
    if predicted[i] == data5_ytest[i]:
        correct += 1

print("For data 5: ")
print("Accuracy by SVM predict function (linear kernel): ", correct/(len(predicted)))

w = model.coef_
b = model.intercept_

correct = 0
for i in range(len(data5_xtest)):
    if data5_ytest[i] == custom_predict_linear(data5_xtest[i], w, b):
        correct += 1

print("Accuracy by implemented predict function (linear kernel): ", correct/(len(data5_xtest)))


model = svm.SVC(kernel='rbf', gamma='auto')
model.fit(data5_xtrain, data5_ytrain)
predicted = model.predict(data5_xtest)

correct = 0
for i in range(len(predicted)):
    if predicted[i] == data5_ytest[i]:
        correct += 1

print("Accuracy by SVM predict function (rbf kernel): ", correct/(len(predicted)))
# print(model.dual_coef_)
# print(model.support_vectors_)
# print(model.intercept_)


# model = svm.SVC(kernel="precomputed", gamma='auto')
# gm = rbf_kernel(data4_xtrain, data4_xtrain)
# model.fit(gm, data4_ytrain)
# predicted = model.predict(rbf_kernel(data4_xtest, data4_xtrain))
# From inbuilt rbf
model = svm.SVC(kernel=rbf_kernel, gamma="auto")
model.fit(data5_xtrain, data5_ytrain)

y_alpha = model.dual_coef_
k_i = model.support_
rho = model.intercept_

# print(model.support_)
# print(model.support_.shape)
# print("y alpha shape: ", y_alpha.shape, "k_i shape: ", k_i.shape, "number: ", model.n_support_)

correct = 0
for i in range(len(data5_xtest)):
    if custom_predict_rbf(data5_xtest[i], y_alpha, k_i, rho, data5_xtrain) == data5_ytest[i]:
        correct += 1

print("Accuracy by implemented predict function (implemented rbf kernel): ", correct/(len(predicted)))
