import os

# ReLU used as an example function
scriptpath = os.getcwd() + "/gen_data_relu/"

digit_array = list(range(10))
digit_combinations = []
for i in digit_array:
    for j in digit_array[1:]:
        if i < j:
            temparr = [i, j]
            digit_combinations.append(temparr)

patharr = []
for combination in digit_combinations:
    try:
        os.mkdir(scriptpath + f"job_{combination[0]}{combination[1]}")
    except:
        pass
    patharr.append(
        f"/storage/eng/esuwrv/gen_data_relu/job_{combination[0]}{combination[1]}\n")
    filename = "/script.py"
    temp_path = scriptpath + f"job_{combination[0]}{combination[1]}" + filename
    file = open(temp_path, 'w')
    file.write(f"""import numpy as np
from numpy import genfromtxt
from PIL import Image
from keras.datasets import mnist
import sys
sys.path.append("../")
from _layers import DenseL2, Sigmoid, Tanh, ReLU
from _functions import *
import time
import os

start_time = time.time()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
accuracy_arr = []
activation_arr = []
for i in range(100):
    _x_train, _y_train = preprocess_data(
        x_train, y_train, filter_arr={combination}, resizing_factor=3)
    _x_test, _y_test = preprocess_data(
        x_test, y_test, filter_arr={combination}, resizing_factor=3)
    network = [
        DenseL2(9, 1),
        ReLU(),
        DenseL2(1, 2),
        ReLU()
    ]
    train(network, mse, mse_prime, _x_train,
            _y_train, epochs=100, learning_rate=0.1)
    acc, _ = get_accuracy(network, _x_test, _y_test)
    accuracy_arr.append(acc)
    activation_arr.append(get_max_activation(network, _x_test))
np.save(
    "/storage/eng/esuwrv/gen_data_relu/readings/{combination[0]}_{combination[1]}_accuracies.npy", accuracy_arr)
np.save(
    "/storage/eng/esuwrv/gen_data_relu/readings/{combination[0]}_{combination[1]}_activations.npy", activation_arr)

end_time = time.time()
print(end_time - start_time)
""")
    file.close()


with open(scriptpath + '/dirs', 'w') as file:
    for item in patharr:
        file.write(item)
