import geneticApi as api
import random
import numpy as np
from random import randint


def fitness(data, label, zero_index, answer):
    dd = data
    for e in range(20):
        dd[zero_index + e] = answer[e]
    label_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label_arr[label] = 1
    label_classifier = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # TODO : با کلسیفایر باید این آرایه رو بدست بیاریم

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
    for j in range(50):
        cr = []
        for element in range(20):
            cr.append(randint(0, 127))
        pop.append(cr)
        ev.append(fitness(data, label, zero_index, cr))
    pop = np.asarray(pop)
    ev = np.asarray(ev)
    for h in range(4000):
        best = np.argmax(ev)
        second_best = randint(0, 49)
        worst = np.argmin(ev)
        child = cross_over(pop[best], pop[second_best], randint(0, 19))
        child = mutation(child, randint(0, 100), randint(0, 19))
        pop[worst] = child
        ev[worst] = fitness(data, label, zero_index, child)
    arg_max = ev[np.argmax(ev)]

    for e in range(20):
        data[zero_index + e] = pop[arg_max][e]
    return np.asarray(data)


if __name__ == '__main__':
    t_data, t_label, t_zero_index = api.get_data()
    print(t_data[0], t_label[0], t_zero_index[0])

    for i in range(len(t_data)):
        t_data[i] = correct_data(t_data[i], t_label[i], t_zero_index[i])
