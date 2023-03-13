import numpy as np
import pandas as pd
from keras.datasets import mnist
from numpy import genfromtxt
from PIL import Image


# Main Layer Framework
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # Forward propagation framework
        pass

    def backward(self, output_gradient, learning_rate):
        # Backward propagation framework
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size, weight_scaling=1, bias_scaling=1):
        self.weights = np.random.randn(
            output_size, input_size) * weight_scaling
        self.bias = np.random.randn(output_size, 1) * bias_scaling

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class DenseL2(Layer):
    def __init__(self, input_size, output_size, weight_scaling=1, bias_scaling=1, weight_penalty=0.0001, weight_threshold=0.2):
        self.weights = np.random.randn(
            output_size, input_size) * weight_scaling
        self.bias = np.random.randn(output_size, 1) * bias_scaling
        self.weight_penalty = weight_penalty
        self.weight_threshold = weight_threshold
        self.weight_scaling = weight_scaling
        self.bias_scaling = bias_scaling

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(
            output_gradient, self.input.T) + 2 * self.weight_penalty * self.weights
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        if np.sum(np.abs(self.weights)) < self.weight_threshold:
            self.weights = np.random.randn(np.shape(self.weights)[0], np.shape(
                self.weights)[1]) * self.weight_scaling
            self.bias = np.random.randn(np.shape(self.bias)[0], np.shape(self.bias)[
                                        1]) * self.bias_scaling
        return input_gradient


# Activation Functions
class Tanh(Activation):
    def __init__(self):
        global Tanh

        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class ScaledTanh(Activation):
    def __init__(self):
        global Tanh

        def tanh(x):
            return np.tanh(x) * 0.2

        def tanh_prime(x):
            return 0.2 * (1 - np.tanh(x) ** 2)

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(x, 0)

        def relu_prime(x):
            return x > 0

        super().__init__(relu, relu_prime)


tanh_simulated_activation = genfromtxt("csv/activation.csv", delimiter=',')


class SimulatedTanh(Activation):
    def __init__(self):
        def simulated_tanh(x):
            output = []
            for value in x:
                coef = tanh_simulated_activation[:, 0].tolist().index(
                    round(float(value), 2))
                output.append(tanh_simulated_activation[coef, 1])
            output = np.array(output)
            output = output.reshape(x.shape)
            return output

        def simulated_tanh_prime(x):
            output = []
            for value in x:
                coef = tanh_simulated_activation[:, 0].tolist().index(
                    round(float(value), 2))
                output.append(
                    tanh_simulated_activation[coef+1, 1] - tanh_simulated_activation[coef-1, 1])
            output = np.array(output)
            output = output.reshape(x.shape)
            return output

        super().__init__(simulated_tanh, simulated_tanh_prime)


tanh_measured_activation = genfromtxt(
    "csv/activation_processed.csv", delimiter=',')


class MeasuredTanh(Activation):
    def __init__(self, delta_x=0.05):
        def measured_tanh(x):
            return np.interp(x, tanh_measured_activation[:, 0], tanh_measured_activation[:, 1])

        def measured_tanh_prime(x):
            delta_y = np.interp(x + delta_x, tanh_measured_activation[:, 0], tanh_measured_activation[:, 1]) - np.interp(
                x - delta_x, tanh_measured_activation[:, 0], tanh_measured_activation[:, 1])
            return delta_y * (1 / (2 * delta_x))

        super().__init__(measured_tanh, measured_tanh_prime)


# Cost Functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


# Helper Functions
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def ans_arr(Y, vals):
    vals = np.array(vals)
    ans = np.array(np.zeros((1, vals.shape[0])))
    for i in range(Y.shape[0]):
        if Y[i] in vals:
            row = np.zeros((1, vals.shape[0]))
            index = np.where(vals == Y[i])
            row[0, index] = 1
            ans = np.vstack((ans, row))
    return ans[1:, :]


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=False, x_test=None, y_test=None):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            # error
            error += loss(y, output)
            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(x_train)
        if verbose:
            if x_test.shape and y_test.shape:
                acc, _ = get_accuracy(network, x_train, y_train)
                print(f"{e + 1}/{epochs}, error={error}, accuracy={acc}")
            else:
                print(f"{e + 1}/{epochs}, error={error}")


def get_accuracy(network, X, Y):
    ans_arr = []
    value_arr = []
    for x, y in zip(X, Y):
        output = predict(network, x)
        if np.argmax(output) == np.argmax(y):
            ans_arr.append(1)
        else:
            ans_arr.append(0)
        ans = [np.argmax(output), np.argmax(y)]
        value_arr.append(ans)
    return np.sum(ans_arr) / Y.shape[0], np.array(value_arr)


def preprocess_data(x, y, filter_arr, resizing_factor=None, limit=None):
    x = x.astype("float32") / 255
    x = x.reshape(x.shape[0], x.shape[1]**2, 1)
    filtered_x = []
    for i in range(len(y)):
        if y[i] in filter_arr:
            if resizing_factor:
                image_data_square = np.reshape(x[i], (28, 28))
                cropped_image_data = image_data_square[4:24, 4:24]
                image_org = Image.fromarray(cropped_image_data)
                image_shrunk = image_org.resize(
                    (resizing_factor, resizing_factor))
                image_shrunk_data = np.array(image_shrunk)
                image_shrunk_data = image_shrunk_data - \
                    np.min(image_shrunk_data)
                image_shrunk_data = image_shrunk_data / \
                    np.max(image_shrunk_data)
                image_data_flat = image_shrunk_data.flatten()
                filtered_x.append(image_data_flat)
            else:
                filtered_x.append(x[i])
    x = np.array(filtered_x)
    x = x.reshape(x.shape[0], resizing_factor**2, 1)
    y = ans_arr(y, vals=filter_arr)
    y = y.reshape(y.shape[0], len(filter_arr), 1)
    if limit:
        return x[:limit], y[:limit]
    return x, y


# Example initialization
(x_train, y_train), (x_test, y_test) = mnist.load_data()
_x_train, _y_train = preprocess_data(
    x_train, y_train, filter_arr=[0, 1], resizing_factor=3)
_x_test, _y_test = preprocess_data(
    x_test, y_test, filter_arr=[0, 1], resizing_factor=3)
network = [
    Dense(9, 1),
    Tanh(),
    Dense(1, 2),
    Tanh()
]
train(network, mse, mse_prime, _x_train,
      _y_train, epochs=100, learning_rate=0.1)
acc, answers = get_accuracy(network, _x_test, _y_test)
