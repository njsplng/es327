import numpy as np
from keras.datasets import mnist
from PIL import Image


# Supplementary Function
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


# Main Function
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


# Example Function Call
(x_train, y_train), (x_test, y_test) = mnist.load_data()
_x_train, _y_train = preprocess_data(
    x_train, y_train, filter_arr=[2, 8], resizing_factor=3)
_x_test, _y_test = preprocess_data(
    x_test, y_test, filter_arr=[2, 8], resizing_factor=3)
