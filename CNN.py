import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as mtp
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss

# Sources Referred
# https://stackoverflow.com/questions/56675943/what-is-the-meaning-of-parameters-involved-in-torch-nn-conv2d
# https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers/data
# https://colab.research.google.com/drive/1MZBcP4OPesGeHZuTX7x17KqNXTDy7k1c#scrollTo=8MDt5ZC0zEtp
# https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forwardpass(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)

        return x


T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
fashion_train = torchvision.datasets.FashionMNIST('', train=True, transform=T, download=False)
fashion_test = torchvision.datasets.FashionMNIST('', train=False, transform=T, download=False)
fashion_train_dataloader = torch.utils.data.DataLoader(fashion_train, batch_size=200)
fashion_test_dataloader = torch.utils.data.DataLoader(fashion_test, batch_size=200)
print("Fashion_train shape: ", len(fashion_train_dataloader.dataset))
print("Fashion_test shape: ", len(fashion_test_dataloader.dataset))


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

iteration = []
loss_train = []
loss_test = []
loss_train_svm = []
loss_test_svm = []
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for epoch in range(10):
    print("Epoch: ", epoch+1)
    loss_per_epoch_train = 0.0
    train_X = np.empty((0, 10))
    train_Y = np.empty((0, ))

    for (features, labels) in fashion_train_dataloader:
        # print(type(labels))
        # print(labels)
        # print(labels.shape)
        optimizer.zero_grad()
        labels_out = model.forwardpass(features)
        loss = criterion(labels_out, labels)
        loss.backward()
        optimizer.step()

        # For CNN
        loss_per_epoch_train += loss.item()
        # correct_train += (torch.max(labels_out, 1)[1].view(200) == labels).sum().item()

        # For SVM
        fv = labels_out.detach().numpy()
        # print(train_X.shape, fv.shape)
        train_X = np.append(train_X, fv, axis=0)
        # print(train_Y.shape, labels.numpy().shape)
        train_Y = np.append(train_Y, labels.numpy(), axis=0)

    loss_per_epoch_test = 0.0
    # correct_test = 0

    test_X = np.empty((0, 10))
    test_Y = np.empty((0, ))
    for (features, labels) in fashion_test_dataloader:
        labels_out = model.forwardpass(features)
        loss = criterion(labels_out, labels)
        loss_per_epoch_test += loss.item()
        # correct_test += (torch.max(labels_out, 1)[1].view(200) == labels).sum().item()

        # For SVM
        fv = labels_out.detach().numpy()
        test_X = np.append(test_X, fv, axis=0)
        test_Y = np.append(test_Y, labels.numpy(), axis=0)

    # print("Loss per epoch train: ", loss_per_epoch_train/300)
    # print("Accuracy per epoch train: ", correct_train*100/60000)
    # print("Loss per epoch test: ", loss_per_epoch_test/50)
    # print("Accuracy per epoch test: ", correct_test*100/10000)

    iteration.append(epoch+1)
    loss_train.append(loss_per_epoch_train/300)
    # accuracy_train.append(correct_train/60000)
    loss_test.append(loss_per_epoch_test/50)
    # accuracy_test.append(correct_test/10000)

    # For SVM
    model_svm = svm.SVC(kernel='rbf', decision_function_shape="ova", gamma="auto")
    model_svm.fit(train_X, train_Y)
    # y_predict_train = model_svm.predict(train_X)
    # as_train = accuracy_score(y_predict_train, train_Y)
    # print("Train accuracy score (SVM): ", as_train*100)
    # accuracy_train_SVM.append(as_train)

    hinge = model_svm.decision_function(train_X)
    # print("Train SVM Loss: ", hinge_loss(train_Y, hinge, labels=classes))
    loss_train_svm.append(hinge_loss(train_Y, hinge, labels=classes))
    # y_predict_test = model_svm.predict(test_X)
    # as_test = accuracy_score(y_predict_test, test_Y)
    # print("Test accuracy score (SVM) :", as_test*100)
    # accuracy_test_SVM.append(as_test)

    hinge = model_svm.decision_function(test_X)
    # print("Test SVM Loss: ", hinge_loss(test_Y, hinge, labels=classes))
    loss_test_svm.append(hinge_loss(test_Y, hinge, labels=classes))

##########################################################################

labels_predict_train = torch.zeros(0, dtype=torch.long, device='cpu')
labels_train = torch.zeros(0, dtype=torch.long, device='cpu')

labels_predict_test = torch.zeros(0, dtype=torch.long, device='cpu')
labels_test = torch.zeros(0, dtype=torch.long, device='cpu')

correct_train = 0
for (features, labels) in fashion_train_dataloader:
    outputs = model.forwardpass(features)
    _, preds = torch.max(outputs, 1)

    correct_train += (torch.max(outputs, 1)[1].view(200) == labels).sum().item()
    labels_predict_train = torch.cat([labels_predict_train, preds.view(-1).cpu()])
    labels_train = torch.cat([labels_train, labels.view(-1).cpu()])


correct_test = 0
for (features, labels) in fashion_test_dataloader:
    outputs = model.forwardpass(features)
    _, preds = torch.max(outputs, 1)

    correct_test += (torch.max(outputs, 1)[1].view(200) == labels).sum().item()
    labels_predict_test = torch.cat([labels_predict_test, preds.view(-1).cpu()])
    labels_test = torch.cat([labels_test, labels.view(-1).cpu()])

print("Accuracy of CNN on Training Data: ", correct_train*100/60000)
print("Accuracy of CNN on Test Data: ", correct_test*100/10000)

# Confusion matrix
cm = confusion_matrix(labels_predict_train.numpy(), labels_train.numpy())
plot_confusion_matrix(conf_mat=cm)
mtp.title("CNN Confusion matrix: Training Set")
mtp.show()

cm = confusion_matrix(labels_predict_test.numpy(), labels_test.numpy())
plot_confusion_matrix(conf_mat=cm)
mtp.title("CNN Confusion matrix: Test Set")
mtp.show()

mtp.plot(iteration, loss_train)
mtp.xlabel("Epoch")
mtp.ylabel("Loss")
mtp.title("Loss vs. epoch (CNN): Training Set")
mtp.show()

mtp.plot(iteration, loss_test)
mtp.xlabel("Epoch")
mtp.ylabel("Loss")
mtp.title("Loss vs. epoch (CNN): Test Set")
mtp.show()

##########################################################################

train_X = np.empty((0, 10))
train_Y = np.empty((0, ))
for (features, labels) in fashion_train_dataloader:
    fv = model.forwardpass(features).detach().numpy()
    train_X = np.append(train_X, fv, axis=0)
    train_Y = np.append(train_Y, labels.numpy(), axis=0)


test_X = np.empty((0, 10))
test_Y = np.empty((0, ))
for (features, labels) in fashion_test_dataloader:
    fv = model.forwardpass(features).detach().numpy()
    test_X = np.append(test_X, fv, axis=0)
    test_Y = np.append(test_Y, labels.numpy(), axis=0)

model_svm = svm.SVC(kernel='rbf')
model_svm.fit(train_X, train_Y)
y_predict_train = model_svm.predict(train_X)
print("Accuracy of SVM (rbf kernel) on Training data:", accuracy_score(train_Y, y_predict_train)*100)

cm = confusion_matrix(y_predict_train, train_Y)
plot_confusion_matrix(conf_mat=cm)
mtp.title("SVM (rbf kernel) Confusion matrix: Training Set")
mtp.show()

y_predict_test = model_svm.predict(test_X)
print("Accuracy of SVM (rbf kernel) on Test data:", accuracy_score(test_Y, y_predict_test)*100)

cm = confusion_matrix(y_predict_test, test_Y)
plot_confusion_matrix(conf_mat=cm)
mtp.title("SVM (rbf kernel) Confusion matrix: Test set")
mtp.show()

mtp.plot(iteration, loss_train_svm)
mtp.xlabel("Epoch")
mtp.ylabel("Loss")
mtp.title("Loss vs. epoch (SVM): Training Set")
mtp.show()

mtp.plot(iteration, loss_test_svm)
mtp.xlabel("Epoch")
mtp.ylabel("Loss")
mtp.title("Loss vs. epoch (SVM): Test Set")
mtp.show()

#######################################################################################



