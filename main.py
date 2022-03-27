import math
import matplotlib.pyplot as plt
from itertools import combinations


class neuron:
    def __init__(self):
        self.present_weights = [0, 0, 0, 0, 0]

    def threshold_weight_correction(self, error, values_vector):
        training_norm = 0.3
        new_weights = [0, 0, 0, 0, 0]
        for index in range(0, 5):
            new_weights[index] = training_norm * error * values_vector[index]
            new_weights[index] += self.present_weights[index]
            if new_weights[index] == 0:
                new_weights[index] = 0
        self.present_weights = new_weights

    def sigmoid_threshold_weight_correction(self, error, values_vector, out):
        training_norm = 0.3
        new_weights = [0, 0, 0, 0, 0]
        for index in range(0, 5):
            sigmoid = out * (1 - out)
            new_weights[index] = training_norm * error * values_vector[index] * sigmoid
            new_weights[index] += self.present_weights[index]
            if new_weights[index] == 0:
                new_weights[index] = 0
        self.present_weights = new_weights

    def net_calculation(self, values_vector):
        net = 0
        for index in range(0, 5):
            net += self.present_weights[index] * values_vector[index]
        return net

    def neuron_boolean_function(self, values_vector):
        net = self.net_calculation(values_vector)
        if net >= 0:
            return int(1)
        else:
            return int(0)

    def sigmoid_neuron_boolean_function(self, values_vector):
        net = self.net_calculation(values_vector)
        out = 1 / (1 + math.exp(-net))
        if out >= 0.5:
            return out, int(1)
        else:
            return out, int(0)


def real_boolean_function(values_vector):
    out = 0
    if values_vector[0] == 1 or values_vector[1] == 1 or values_vector[3] == 1:
        out = 1
    if values_vector[2] == 0:
        out = 0
    return out


def take_vector(number):
    out_vector = [1]
    number = bin(number)
    number = number[2:]
    for j in range(0, 4):
        if len(number) != 0:
            out_vector.append(int(number[-1]))
            number = number[:-1]
        else:
            out_vector.append(int(0))
    out_vector.reverse()
    return out_vector


def print_epoch(epoch_number, count_errors, out_vector, weights):
    str_out_vector = '('
    for it in out_vector:
        str_out_vector += str(it)
        str_out_vector += ', '
    str_out_vector = str_out_vector[:-2]
    str_out_vector += ')'
    str_present_weights = '('
    for it in weights:
        str_present_weights += str(round(it, 2))
        str_present_weights += ', '
    str_present_weights = str_present_weights[:-2]
    str_present_weights += ')'
    print(epoch_number.center(11, ' '), str_present_weights.center(33, ' '),
          str_out_vector.center(32, ' '), str(count_errors).center(15, ' '))


def threshold_neuron_function():
    plot_epoch_number = []
    plot_count_errors = []
    epoch_number = 0
    my_neuron = neuron()
    flag = True
    print(
        'Номер эпохи           Вектор весов                            Выходной вектор                Суммарная ошибка')
    while flag:
        out_vector = []
        epoch_number += 1
        count_errors = 0
        weights = my_neuron.present_weights
        for i in range(0, 16):
            values_vector = take_vector(i)
            real_boolean_out = real_boolean_function(values_vector)
            neuron_boolean_out = my_neuron.neuron_boolean_function(values_vector)
            out_vector.append(neuron_boolean_out)
            error = real_boolean_out - neuron_boolean_out
            if error != 0:
                count_errors += 1
            my_neuron.threshold_weight_correction(error, values_vector)
        if count_errors == 0:
            flag = False
        print_epoch(str(epoch_number), count_errors, out_vector, weights)
        plot_epoch_number.append(epoch_number)
        plot_count_errors.append(count_errors)
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.plot(plot_epoch_number, plot_count_errors)
    plt.show()


def sigmoid_neuron_function():
    plot_epoch_number, plot_count_errors = [], []
    epoch_number = 0
    my_neuron = neuron()
    flag = True
    print(
        'Номер эпохи           Вектор весов                            Выходной вектор                Суммарная '
        'ошибка')
    while flag:
        out_vector = []
        epoch_number += 1
        count_errors = 0
        weights = my_neuron.present_weights
        for i in range(0, 16):
            values_vector = take_vector(i)
            real_boolean_out = real_boolean_function(values_vector)
            out, sigmoid_neuron_boolean_out = my_neuron.sigmoid_neuron_boolean_function(values_vector)
            out_vector.append(sigmoid_neuron_boolean_out)
            error = real_boolean_out - sigmoid_neuron_boolean_out
            if error != 0:
                count_errors += 1
            my_neuron.sigmoid_threshold_weight_correction(error, values_vector, out)
        if count_errors == 0:
            flag = False
        print_epoch(str(epoch_number), count_errors, out_vector, weights)
        plot_epoch_number.append(epoch_number)
        plot_count_errors.append(count_errors)
    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.plot(plot_epoch_number, plot_count_errors)
    plt.show()


def test_subset(my_neuron):
    for i in range(0, 16):
        values_vector = take_vector(i)
        real_boolean_out = real_boolean_function(values_vector)
        out, sigmoid_neuron_boolean_out = my_neuron.sigmoid_neuron_boolean_function(values_vector)
        error = real_boolean_out - sigmoid_neuron_boolean_out
        if error != 0:
            return False
    return True


def try_subset(subset):
    epoch_number = 0
    flag = True
    my_neuron = neuron()
    while flag and epoch_number < 100:
        out_vector = []
        epoch_number += 1
        count_errors = 0
        for vec in subset:
            values_vector = []
            for it_first in vec:
                values_vector.append(int(it_first))
            real_boolean_out = real_boolean_function(values_vector)
            out, sigmoid_neuron_boolean_out = my_neuron.sigmoid_neuron_boolean_function(values_vector)
            out_vector.append(sigmoid_neuron_boolean_out)
            error = real_boolean_out - sigmoid_neuron_boolean_out
            if error != 0:
                count_errors += 1
            my_neuron.sigmoid_threshold_weight_correction(error, values_vector, out)
        if count_errors == 0 and test_subset(my_neuron):
            flag = False
    if epoch_number < 100:
        return epoch_number, my_neuron.present_weights, True
    else:
        return None, None, False


def search_min_subset():
    all_values_vector = []
    min_subset = [[]]
    for i in range(0, 17):
        zero_vector = 0
        min_subset.append(zero_vector)
    for i in range(0, 16):
        all_values_vector.append(take_vector(i))
    for j in range(len(all_values_vector) + 1):
        if j == 0:
            continue
        subset_list = list(combinations(all_values_vector, j))
        for subset in subset_list:
            epoch_number, weights, is_try_subset = try_subset(subset)
            if is_try_subset:
                print("Минимальный набор векторов:")
                print(subset)
                print("Количество эпох:", epoch_number)
                print("Синаптические коэффиценты:")
                print(weights)
                return


if __name__ == '__main__':
    threshold_neuron_function()
    sigmoid_neuron_function()
    search_min_subset()
