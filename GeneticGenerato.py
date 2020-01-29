import geneticApi as api
import random
import numpy as np
from random import randint
import funcs
from keras.models import load_model

def fitness(data, label, zero_index, answer):
    out = np.zeros(5)
    if label == 3:
        out[0] = 1
    elif label == 5:
        out[1] = 1
    elif label == 6:
        out[2] = 1
    elif label == 7:
        out[3] = 1
    elif label == 9:
        out[4] = 1

    dd = data
    for e in range(20):
        dd[zero_index + e] = answer[e]
    label_arr = out
    model = load_model('./model.h5')
    g = [dd]
    res = model.predict(funcs.extract(g))
    label_classifier = res[0]  # TODO : با کلسیفایر باید این آرایه رو بدست بیاریم

    score = 0
    for f in range(len(label_arr)):
        d = label_arr[f] - label_classifier[f]
        if d < 0:
            d *= -1
        score += d
    return 5 - score


def cross_over(parent1, parent2, location):
    res = []
    for i in range(20):
        if i < location:
            res.append(parent1[i])
        else:
            res.append(parent2[i])
    return np.asarray(res)


def mutation(child, probability, location):
    if probability < 10:
        child[location] = randint(0, 127)
    return child


def correct_data(data, label, zero_index):
    pop = []
    ev = []
    for j in range(20):
        cr = []
        for element in range(20):
            cr.append(randint(0, 127))
        pop.append(cr)
        ev.append(fitness(data, label, zero_index, cr))
    pop = np.asarray(pop)
    ev = np.asarray(ev)
    for h in range(10):
        print(h)
        best = np.argmax(ev)
        second_best = randint(0, 19)
        worst = np.argmin(ev)
        child = cross_over(pop[best], pop[second_best], randint(0, 19))
        child = mutation(child, randint(0, 100), randint(0, 19))
        pop[worst] = child
        ev[worst] = fitness(data, label, zero_index, child)
        print(best, second_best, worst)
    arg_max = np.argmax(ev)

    for e in range(20):
        data[zero_index + e] = pop[arg_max][e]
    return np.asarray(data)


if __name__ == '__main__':
    t_data, t_label, t_zero_index = api.get_data()
    print(t_data[0], t_label[0], t_zero_index[0])

    for i in range(len(t_data)):
        print("data =>")
        t_data[i] = correct_data(t_data[i], t_label[i], t_zero_index[i])

    model = load_model('./model.h5')

    output_data = []
    for item in t_label:
        out = np.zeros(5)
        if item == 3:
            out[0] = 1
        elif item == 5:
            out[1] = 1
        elif item == 6:
            out[2] = 1
        elif item == 7:
            out[3] = 1
        elif item == 9:
            out[4] = 1
        output_data.append(out)
    output_data = np.asarray(output_data)
    f = funcs.extract(t_data)
    res = model.evaluate(f, output_data)
    print(res)
    print(model.metrics_names)

    res = model.predict(f)

    for i, item in enumerate(res):
        idxMax = np.argmax(item)
        predict = 0

        if idxMax == 0:
            predict = 3
        elif idxMax == 1:
            predict = 5
        elif idxMax == 2:
            predict = 6
        elif idxMax == 3:
            predict = 7
        elif idxMax == 4:
            predict = 9
        print("predict " + str(predict) + " real " + str(t_label[i]))

