


#Import 
import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import funcs
import keras
from keras import losses
from keras.activations import sigmoid
from matplotlib import pyplot

# for i , item in enumerate(data_test):
    
print('Geting data')
data_train, label_train = funcs.getTraindata()
# data_train = data_train[0:10000]
# label_train = label_train[0:10000]
print('Start NN')
print(data_train[0:10])
print(label_train[0:10])
def custom_loss():

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return (y_true - y_pred) ** 2
   
    # Return a 
    return loss

model = Sequential()
model.add(Dense(7,  activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
# model.add(Dense(48, activation=sigmoid))
# model.add(Dense(96, activation=sigmoid))
model.add(Dense(128,activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','cosine_proximity'])
print("Start fit data")
trainedModel = model.fit(data_train, label_train, epochs=10, batch_size=100)
print("get test data")
# data_test, label_test = funcs.getTestData()
# res = model.evaluate(data_train, y = label_train, batch_size=32, verbose=1)
# print(model.evaluate(data_test, label_test))
# res = model.predict(data_test)

# res = ( res * 100).astype('int16')
model.save('./test-128.h5')
print("Model saved")
# for i in range(res.shape[0]):
#     print(data_test[i])
#     # print(np.argmax(res[i]))
#     print(res[i])
#     print('expected ' + str(label_test[i]))
#     print('-------------------------------------------------')
# print(res.shape)