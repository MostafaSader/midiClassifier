from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import funcs
from collections import Counter

if __name__ == '__main__':
    print('Getting data')
    # data_train, label_train = funcs.get_train_data()
    data_train = funcs.load_data("INPUT_DATA.pkl")
    label_train = funcs.load_data("INPUT_LABELS.pkl")
    print("len of train " + str(len(data_train)))
    print("len of lable " + str(len(label_train)))
    print("===========================")
    # Provide Input Data
    # ========================================================
    input_data = []
    for data_item in data_train:
        mean = np.mean(data_item)

        dt = Counter(data_item)
        ent = np.zeros(129)
        ent[0] = mean
        for idx in range(1,128):
            ent[idx] = dt[idx]
        input_data.append(ent)
    input_data = np.asarray(input_data)
    # print(len(input_data))
    # print(input_data[0])
    # exit()
    
    # Provide Label
    # ========================================================
    output_data = []
    for item in label_train:
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
    # print(output_data)
    # print("Shape of input " + str(input_data.shape))
    # print("Shape of output " + str(output_data.shape))
    print("===========================")
    print('Start NN')
    model = Sequential()
    model.add(Dense(129, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(54, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'cosine_proximity'])
    print("Start fit data")
    trainedModel = model.fit(input_data, output_data, epochs=100, batch_size=1)
    print("get test data")
    model.save('./model.h5')
    print("Model saved")
