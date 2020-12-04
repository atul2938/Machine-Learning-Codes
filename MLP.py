import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as mtp

# Sources Used
# https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/


class NeuralNet:
    def __init__(self, num_layers, act_func, learning_rate, data, label):
        self.numLayers = num_layers
        self.actFunc = act_func
        self.lr = learning_rate
        self.numHidden = len(num_layers)
        self.error = []

        self.bias = {}
        for i in range(self.numHidden):
            self.bias[i] = np.zeros(num_layers[i])
        self.bias[self.numHidden] = np.zeros(label)

        self.weight = {0: np.array([np.random.normal(0, 1, data) for i in range(num_layers[0])]).T * 0.01}
        for i in range(1, self.numHidden):
            self.weight[i] = np.array([np.random.normal(0, 1, num_layers[i-1]) for j in range(num_layers[i])]).T * 0.01
        self.weight[self.numHidden] = np.array([np.random.normal(0, 1, num_layers[self.numHidden-1])
                                                for i in range(label)]).T * 0.01

    def activation(self, x, derivative=False):
        if self.actFunc == 'relu':
            if not derivative:
                return self.relu(x)
            else:
                return self.relu_derivative(x)
        elif self.actFunc == 'linear':
            if not derivative:
                return self.linear(x)
            else:
                return self.linear_derivative(x)
        elif self.actFunc == 'tanh':
            if not derivative:
                return self.tanh(x)
            else:
                return self.tanh_derivative(x)
        else:
            if not derivative:
                return self.sigmoid(x)
            else:
                return self.sigmoid_derivative(x)

    def forward_pass(self, x):
        v = {}
        y = {0: x}
        for i in range(self.numHidden):
            v[i+1] = np.dot(y[i], self.weight[i]) + self.bias[i]
            y[i+1] = self.activation(v[i+1])
        v[self.numHidden+1] = np.dot(y[self.numHidden], self.weight[self.numHidden])
        y[self.numHidden+1] = self.softmax(v[self.numHidden+1])

        return v, y

    # Accepts input data, input labels, batch size and epochs and returns nothing
    def fit(self, data, label, batch_size, epochs):
        for i in range(epochs):
            print("Epoch: ", i+1)
            error_per_epoch = 0.0
            for j in range(len(data)//batch_size):
                x = data[j*batch_size: (j+1)*batch_size, ]
                y = label[j*batch_size: (j+1)*batch_size, ]

                # Forward Pass
                v, a = self.forward_pass(x)

                # backward phase
                e = self.loss_function(a[self.numHidden+1], y)
                # Short for cross entropy activation function with softmax activation function derivative
                error = a[self.numHidden+1] - e
                error_per_epoch += np.sum(np.absolute(error))

                # Output layer
                dw = {self.numHidden: np.dot(a[self.numHidden].T, error)}
                db = {self.numHidden: error}
                lg = {self.numHidden+1: error}

                for k in range(self.numHidden, 0, -1):
                    dy = self.activation(v[k], True)
                    dcost = np.dot(lg[k+1], self.weight[k].T)
                    lg[k] = np.multiply(dy, dcost)
                    dw[k-1] = np.dot(lg[k].T, a[k-1]).T
                    db[k-1] = dcost * dy

                # Update weights and bias
                for k in range(self.numHidden+1):
                    self.bias[k] = self.bias[k] - self.lr * db[k].sum(axis=0)
                    self.weight[k] = self.weight[k] - self.lr * dw[k]

                # print("Accuracy score for epoch ", i, " - batch ", j, ": ", self.score(x, y)*100)

            self.error.append(error_per_epoch/len(data))

    # Accepts input data and returns class wise probabilities for the given data
    def predict(self, x):
        v, y = self.forward_pass(x)
        return y[self.numHidden+1]

    # Accepts input data and their labels and returns accuracy.
    def score(self, x, y):
        y_predict = self.predict(x)
        N = len(y)
        n = 0
        y = y.astype(int)
        for i in range(N):
            p_max = 0
            index_max = 0
            for j in range(10):
                if y_predict[i][j] > p_max:
                    p_max = y_predict[i][j]
                    index_max = j
            if index_max == y[i]:
                n = n+1
        return n/N

    # cross entropy loss function
    @staticmethod
    def loss_function(y_predict, y):
        e = np.zeros(y_predict.shape)
        y = y.astype(int)
        for i in range(len(y)):
            e[i][y[i]] = 1

        return e

    @staticmethod
    def relu(x):
        act = np.ones((len(x), len(x[0])))
        for i in range(len(x)):
            for j in range(len(x[0])):
                act[i][j] = max(0, x[i][j])
        return act

    # Referred from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    @staticmethod
    def relu_derivative(x):
        act = np.ones((len(x), len(x[0])))
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i][j] > 0:
                    act[i][j] = 1
                else:
                    act[i][j] = 0
        return act

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones(x.shape)

    def tanh(self, x):
        return 2*self.sigmoid(2*x)-1

    # Referred from https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    def tanh_derivative(self, x):
        return 1 - np.square(self.tanh(x))

    # Referred from https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    @staticmethod
    def softmax(x):
        # e_x = np.exp(x.T - np.max(x, axis=-1))
        # return (e_x / e_x.sum(axis=0)).T
        # print("shape: ", x.shape)
        z = x - np.max(x, axis=1)[:, np.newaxis]
        sm = np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]
        return sm


# Load training data
train_data = np.loadtxt('mnist_train.csv', delimiter=',')
np.random.shuffle(train_data)
train_X = np.delete(train_data, 784, 1)
train_X = train_X[:20000, :]
train_Y = train_data[:20000, 784]
# train_X = (train_X - train_X.mean(axis=0))/train_X.std(axis=0)
print("Train feature shape: ", train_X.shape)
print("Train label shape: ", train_Y.shape)

# Initialize the models
mlp_relu = NeuralNet([256, 128, 64], 'relu', 0.1, 784, 10)
mlp_identity = NeuralNet([256, 128, 64], 'linear', 0.1, 784, 10)
mlp_sigmoid = NeuralNet([256, 128, 64], 'sigmoid', 0.1, 784, 10)
mlp_tanh = NeuralNet([256, 128, 64], 'tanh', 0.1, 784, 10)

# Train the models
mlp_relu.fit(train_X, train_Y, 200, 10)
mlp_identity.fit(train_X, train_Y, 200, 10)
mlp_sigmoid.fit(train_X, train_Y, 200, 10)
mlp_tanh.fit(train_X, train_Y, 200, 10)

# Save the weights
with open("pkl_relu", 'wb') as file:
    pickle.dump(mlp_relu, file)
with open("pkl_identity", 'wb') as file:
    pickle.dump(mlp_identity, file)
with open("pkl_sigmoid", 'wb') as file:
    pickle.dump(mlp_sigmoid, file)
with open("pkl_tanh", 'wb') as file:
    pickle.dump(mlp_tanh, file)

iteration = [i+1 for i in range(10)]

# print("Len relu:", len(iteration), len(mlp_relu.error))
# print(mlp_relu.error[10], mlp_relu.error[11])
mtp.plot(iteration, mlp_relu.error)
mtp.xlabel("Epoch")
mtp.ylabel("Training error")
mtp.title("Training error vs. epoch (relu)")
mtp.show()

# print("Len identity:", len(iteration), len(mlp_identity.error))
# print(mlp_identity.error[10], mlp_identity.error[11])
mtp.plot(iteration, mlp_identity.error)
mtp.xlabel("Epoch")
mtp.ylabel("Training error")
mtp.title("Training error vs. epoch (identity)")
mtp.show()

# print("Len sigmoid:", len(iteration), len(mlp_sigmoid.error))
# print(mlp_sigmoid.error[10], mlp_sigmoid.error[11])
mtp.plot(iteration, mlp_sigmoid.error)
mtp.xlabel("Epoch")
mtp.ylabel("Training error")
mtp.title("Training error vs. epoch (sigmoid)")
mtp.show()

# print("Len tanh:", len(iteration), len(mlp_tanh.error))
# print(mlp_tanh.error[10], mlp_tanh.error[11])
mtp.plot(iteration, mlp_tanh.error)
mtp.xlabel("Epoch")
mtp.ylabel("Training error")
mtp.title("Training error vs. epoch (tanh)")
mtp.show()

# Load text data
test_data = np.loadtxt('mnist_test.csv', delimiter=',')
np.random.shuffle(test_data)
test_X = np.delete(test_data, 784, 1)
test_X = test_X[:2000, :]
test_Y = test_data[:2000, 784]
print("Test feature shape: ", test_X.shape)
print("Test label shape: ", test_Y.shape)

print("Accuracy score for mlp (relu): ", mlp_relu.score(test_X, test_Y)*100)
print("Accuracy score for mlp (identity): ", mlp_identity.score(test_X, test_Y)*100)
print("Accuracy score for mlp (sigmoid): ", mlp_sigmoid.score(test_X, test_Y)*100)
print("Accuracy score for mlp (tanh): ", mlp_tanh.score(test_X, test_Y)*100)


# Sklearn classifiers initialization
model_relu = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='sgd', learning_rate_init=0.1)
model_identity = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='identity', solver='sgd', learning_rate_init=0.1)
model_sigmoid = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='logistic', solver='sgd', learning_rate_init=0.1)
model_tanh = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='tanh', solver='sgd', learning_rate_init=0.1)

# Training the classifiers
model_relu.fit(train_X, train_Y)
model_identity.fit(train_X, train_Y)
model_sigmoid.fit(train_X, train_Y)
model_tanh.fit(train_X, train_Y)

print("Accuracy score for sklearn mlp (relu): ", model_relu.score(test_X, test_Y)*100)
print("Accuracy score for sklearn mlp (identity): ", model_identity.score(test_X, test_Y)*100)
print("Accuracy score for sklearn mlp (sigmoid): ", model_sigmoid.score(test_X, test_Y)*100)
print("Accuracy score for sklearn mlp (tanh): ", model_tanh.score(test_X, test_Y)*100)

