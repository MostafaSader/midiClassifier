from mido import MidiFile
import numpy as np
import os
import pandas as pd


def get_features_from_midi_file_name(file_name):
    temp = get(os.path.join('./MIDI_Genres/train_set', file_name))
    return temp


def get_train_data():
    data = []
    label = []
    file_list = pd.read_csv('trainLabels.txt').values
    for row in file_list:
        label = row[1]
        data = get_features_from_midi_file_name(row[0])
        label.append(label)
        data.append(data)
    label = np.asarray(label)
    data = np.asarray(data)
    return data, label


def get(file):
    mid = MidiFile(file)
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                if msg.velocity != 0:
                    vector.append(msg.note)
    return vector
