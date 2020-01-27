import mido
from mido import MidiFile
import numpy as np
import os
import pandas as pd
# Constant
directory_train = './data/train/'
directory_test_label = './data/validation/groundTruth/'
directory_test = './data/validation/query/'
directory_test_final = './data/test'


def get_train_data():
    return pd.read_csv('trainLabels.txt')



def get_test_data():
    data = []
    label = []
    label_query = []
    for file in os.listdir(directory_test):
        mid = MidiFile(os.path.join(directory_test, file))
        temp = get(mid)
        for i in range(len(temp) - 7):
            # lbl = np.zeros(128)
            data.append(np.array(temp[i:i + 7]))
            label_query.append(temp[i + 7])
            # lbl[temp[i + 7]] = 1
            # label.append(np.array(lbl))
    for file in os.listdir(directory_test_label):
        mid = MidiFile(os.path.join(directory_test_label, file))
        temp = get(mid)
        for i in range(len(temp) - 8):
            # lbl = np.zeros(128)
            # data.append(np.array(temp[i:i + 7]))
            # lbl[temp[i + 7]] = 1
            # label.append(np.array(lbl))

            label.append(temp[i + 7])
    return np.asarray(data), np.asarray(label), np.asarray(label_query)


def getFinalTest():
    data = []
    label = []
    for file in os.listdir(directory_test):
        mid = MidiFile(os.path.join(directory_test, file))
        temp = get(mid)
        for i in range(len(temp) - 7):
            lbl = np.zeros(128)
            data.append(np.array(temp[i:i + 7]))
            lbl[temp[i + 7]] = 1
            label.append(np.array(lbl))
            # label.append(temp[i + 7])

    return np.asarray(data), np.asarray(label);
def get(mid):
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vector.append(msg.note)
    return vector
