from mido import MidiFile
import numpy as np
import os
import pandas as pd
import _pickle as pkl


def get_features_from_midi_file_name(file_name):
    temp = get(os.path.join('./MIDI_Genres/train_set', file_name))
    return temp

def get_test_from_midi_file_name(file_name):
    temp = get(os.path.join('./MIDI_Genres/python_test_set', file_name))
    return temp

def save_data(data,name):
    pkl.dump(data, open( name, "wb" ) )

def load_data(name):
    data = pkl.load(open( name, "rb" ))
    return data

def get_train_data():
    i = 1
    files = []
    labels = []
    file_list = pd.read_csv('trainLabels.txt').values
    for row in file_list:
        print(i)
        print("=========")
        print(row[0])
        print("=========")
        try:
            data = get_features_from_midi_file_name(row[0])
        except FileNotFoundError as identifier:
            print("NOT FOUND")
            continue
        finally:
            i = i + 1

        labels.append(row[1])
        files.append(data)
    # labels = np.asarray(labels)
    # files = np.asarray(data)

    save_data(files,"INPUT_DATA.pkl")
    save_data(labels,"INPUT_LABELS.pkl")
    print("Data Saved To files")
    exit()
    return files, labels

def get_test_data():
    i = 1
    files = []
    labels = []
    file_list = pd.read_csv('testLabel.txt').values
    for row in file_list:
        if i == 10:
            break
        i = i + 1
        data = get_test_from_midi_file_name(row[0])
        labels.append(row[1])
        files.append(np.asarray(data))
    labels = np.asarray(labels)
    files = np.asarray(files)
    return files, labels


def get(file):
    mid = MidiFile(file)
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                if msg.velocity != 0:
                    vector.append(msg.note)
    return vector


def ext(data):
    f = [data[np.argmax(data)], data[np.argmin(data)]]

    histogram = np.zeros(128)
    mean = 0
    last = data[0]
    max_shib_p = 0
    max = 0
    min = 130
    max_shib_n = 0
    for d in data:
        if max < d:
            max = d
        if min > d:
            min =d
        if d - last > 0:
            if abs(d - last) > max_shib_p:
                max_shib_p = abs(d - last)
        if d - last < 0:
            if abs(d - last) > max_shib_n:
                max_shib_n = abs(d - last)
        last = d
        mean += d
        histogram[d] += 1
    f.append(max_shib_p)
    f.append(max_shib_n)
    s = mean / len(data)
    f.append(s)
    f.append(max)
    f.append(min)
    for h in histogram:
        f.append(h)

    return np.asarray(f)


def extract(input_data):
    ex = []
    for i in range (len(input_data)):
        ex.append(ext(input_data[i]))
    return np.asarray(ex)
