from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as mtp
import scikitplot as skplt


def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


# data_batch_1
dictionary = unpickle("data_batch_1")
print(dictionary.keys())
# print(dictionary)
label = dictionary[b'labels']
data = dictionary[b'data']
# print(Counter(label))
# print(len(label))
# print(data.shape)
# print(len(set(label)))

dictionary_test = unpickle("test_batch")
# model = svm.SVC(kernel='linear', decision_function_shape='ovo', gamma='auto')
# model.fit(data[:2000, :], label[:2000])
# predicted = model.predict(data)
# print(accuracy_score(label, predicted))
t_label = dictionary_test[b'labels']
t_data = dictionary_test[b'data']

# Sampling Test Set
test_data = np.zeros(shape=(200, 3072))
test_label = np.zeros(200)
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
x = 0
for i in range(10000):
    if count[t_label[i]] < 20:
        test_data[x] = t_data[x]
        test_label[x] = t_label[i]
        count[t_label[i]] += 1
        x += 1

# print(test_data.shape, test_data)
# print(test_label.shape, test_label)

# Sampling Data Set
data_all = []
count = []
for i in range(10):
    data_all.append(np.zeros(shape=(100, 3072)))
    count.append(0)


for i in range(10000):
    if count[label[i]] < 100:
        data_all[label[i]][count[label[i]]] = data[i]
        count[label[i]] += 1

sample_data = np.zeros(shape=(1000, 3072))
sample_label = np.zeros(1000)
for i in range(1000):
    sample_data[i] = data_all[i % 10][i//10]
    sample_label[i] = i % 10

# print(type(sample_data))
# print(type(sample_label))

# for i in range(10000):
#     if(label[i]==2):
#         print(data[i])
#         break

# 5 fold
partition = 200
for i in range(5):
    print("\nFor fold: ", (i+1))

    data_val = sample_data[i*partition: (i+1)*partition, :]
    label_val = sample_label[i*partition: (i+1)*partition]
    data_train = np.delete(sample_data, slice(i*partition, (i+1)*partition), 0)
    label_train = np.delete(sample_label, slice(i*partition, (i+1)*partition), 0)

    # Model 1
    model = svm.SVC(kernel='linear', decision_function_shape="ova", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with no kernel: 1-vs-all")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-all - Confusion matrix for validation set")
    mtp.show()

    # pv = label_binarize(p_val, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # skplt.metrics.plot_roc(label_val, pv)
    # mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-all - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-all - Confusion matrix for test set")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-all - ROC Curves for test set")
    mtp.show()

    # Model 2
    model = svm.SVC(kernel='linear', decision_function_shape="ovo", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with no kernel: 1-vs-1")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-1- Confusion matrix for validation set")
    mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-1 - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-1 - Confusion matrix for test set")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with no kernel: 1-vs-1 - ROC Curves for test set")
    mtp.show()

    # Model 3
    model = svm.SVC(kernel='rbf', decision_function_shape="ova", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with rbf kernel: 1-vs-all")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-all - Confusion matrix for validation set")
    mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-all - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-all - Confusion matrix for test test")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-all - ROC Curves for test set")
    mtp.show()

    # Model 4
    model = svm.SVC(kernel='rbf', decision_function_shape="ovo", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with rbf kernel: 1-vs-1")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-1 - Confusion matrix for validation set")
    mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-1 - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-1 - Confusion matrix for test set")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with rbf kernel: 1-vs-1 - ROC Curves for test set")
    mtp.show()

    # Model 5
    model = svm.SVC(kernel='poly', degree=2, decision_function_shape="ova", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with quadratic polynomial kernel: 1-vs-all")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-all - Confusion matrix for validation set")
    mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-all - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-all - Confusion matrix for test set")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-all - ROC Curves for test set")
    mtp.show()

    # Model 6
    model = svm.SVC(kernel='poly', degree=2, decision_function_shape="ovo", gamma="auto", probability=True)
    model.fit(data_train, label_train)

    p_train = model.predict(data_train)
    print("SVM with quadratic polynomial kernel: 1-vs-1")
    # print("Accuracy on training set: ", accuracy_score(label_train, p_train))

    p_val = model.predict(data_val)
    print("Accuracy on validation set: ", accuracy_score(label_val, p_val))
    cm = confusion_matrix(label_val, p_val)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-1 - Confusion matrix for validation set")
    mtp.show()

    p_val_proba = model.predict_proba(data_val)
    skplt.metrics.plot_roc(label_val, p_val_proba, title="Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-1 - ROC Curves for validation set")
    mtp.show()

    p_test = model.predict(test_data)
    print("Accuracy on test set: ", accuracy_score(test_label, p_test))
    cm = confusion_matrix(test_label, p_test)
    plot_confusion_matrix(conf_mat=cm)
    mtp.title("Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-1 - Confusion matrix for test set")
    mtp.show()

    p_test_proba = model.predict_proba(test_data)
    skplt.metrics.plot_roc(test_label, p_test_proba, title="Fold: " + str(i+1) + " - SVM with quad polynomial kernel: 1-vs-1 - ROC Curves for test set")
    mtp.show()


