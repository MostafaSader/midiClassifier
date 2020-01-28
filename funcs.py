from mido import MidiFile
import numpy as np
import os
import pandas as pd


def get_features_from_midi_file_name(file_name):
    temp = get(os.path.join('./MIDI_Genres/train_set', file_name))
    return temp


def get_train_data():
    i = 1
    files = []
    labels = []
    file_list = pd.read_csv('trainLabels.txt').values
    for row in file_list:
        if i == 10:
            break
        print(i)
        print("=========")
        print(row[0])
        print("=========")
        i = i + 1
        data = get_features_from_midi_file_name(row[0])
        labels.append(row[1])
        files.append(data)
    # labels = np.asarray(labels)
    # files = np.asarray(data)
    i = 0
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
