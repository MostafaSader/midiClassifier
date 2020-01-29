import numpy as np
import pandas as pd
import funcs
import os
def get_data():
    label = []
    data = []
    zero_index = []
    file_list = pd.read_csv('testLabel.txt').values
    for file in file_list:
        label.append(file[1])
        path = os.path.join("./MIDI_Genres/python_test_set", file[0])
        d = funcs.get(path)
        data.append(d)
        for i in range(len(d)):
            if d[i] == 0:
                zero_index.append(i)
                break

    return np.asarray(data), np.asarray(label), np.asarray(zero_index)
