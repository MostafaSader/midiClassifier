import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import funcs

if __name__ == '__main__':
    np.random.seed(123)
    print('Getting data')
    data_train, label_train = funcs.get_train_data()
    print('Start NN')
    print(data_train[0:10])
    print(label_train[0:10])
    model = Sequential()
    model.add(Dense(7, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'cosine_proximity'])
    print("Start fit data")
    trainedModel = model.fit(data_train, label_train, epochs=10, batch_size=100)
    print("get test data")

    model.save('./test-128.h5')
    print("Model saved")
