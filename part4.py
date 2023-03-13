(x_train, y_train), (x_test, y_test) = mnist.load_data()
i = 0
while True:
    combination = digit_combinations[random.randint(0, 44)]
    _x_train, _y_train = preprocess_data(
        x_train, y_train, filter_arr=combination, resizing_factor=3)
    _x_test, _y_test = preprocess_data(
        x_test, y_test, filter_arr=combination, resizing_factor=3)
    network = [
        DenseL2(9, 1, weight_penalty=0.0001),
        MeasuredTanh(),
        DenseL2(1, 2, weight_penalty=0.00005),
        MeasuredTanh()
    ]
    train(network, mse, mse_prime, _x_train,
          _y_train, epochs=100, learning_rate=0.1)
    acc, answers = get_accuracy(network, _x_test, _y_test)
    randInt = random.randint(0, _x_test.shape[0])
    test_x = _x_test[randInt, :, :]
    test_y = _y_test[randInt, :, :]
    test_x = np.round(test_x * (255 / 5)) / (255 / 5)
    output = list(predict(network, test_x))
    if acc < 0.8:
        print("Skipped: Accuracy")
        continue
    if abs(output[0] - output[1]) < 0.05:
        print("Skipped: Difference")
        continue
    params = list(get_weights(network, sorted=False))
    for layer in network:
        if hasattr(layer, 'bias'):
            for bias in layer.bias:
                params.append(bias[0])
    params = np.around(params, 2)
    test_x = np.round(test_x, 2)
    output = np.round(output, 2)
    x_display = []
    for val in test_x:
        x_display.append(val[0])
    output_display = []
    for val in output:
        output_display.append(val[0])
    y_display = []
    for val in test_y:
        y_display.append(val[0])
    intermediate = np.round(predict_verbose(network, test_x), 2)
    print("1. Combination value: " + str(combination))
    print("2. Network params: " + str(list(params)))
    print("3. X vals: " + str(x_display))
    print("4. Y predicted vals: " + str(output_display))
    print("5. Y actual vals: " + str(y_display))
    print("6. All activations: " + str(intermediate))
    i += 1
    if i == 10:
        break
