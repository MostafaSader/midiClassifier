
import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import funcs
from keras.optimizers import Adam
from keras import losses
from keras.activations import sigmoid
from matplotlib import pyplot as plt
from keras.models import load_model


model = load_model('./test-128.h5')
print("get test data")

data_test, label_query= funcs.getFinalTest()
# data_test = data_test[0:1000]
# label_query = label_query[0:1000]
res = model.predict(data_test)
print(data_test.shape)
print(data_test[0].shape)

# res = ( res * 100).astype('int16')
# model.save('./test128.h5')

idx = 0
cnt = len(data_test)
dataRes = []
curectedNotes = 0
for i in range(cnt - 6):
    labelQuery = np.argmax(label_query[i])
    dataTest = np.array([data_test[i]])
    res = model.predict(dataTest)
    res = np.argmax(res)
    if abs(res - labelQuery) >= 12:
        data_test[i+1][6] = res
        data_test[i+2][5] = res
        data_test[i+3][4] = res
        data_test[i+4][3] = res
        data_test[i+5][2] = res
        data_test[i+6][1] = res
        curectedNotes = curectedNotes + 1
        print("noise corrected")
        print("data " + str(i) + " changed from " + str(labelQuery) + " to " + str(res))
        dataRes.append(res)
    else:
        dataRes.append(labelQuery)
print("Total currectness : " + str(curectedNotes))
x = []
y = []
for i in range(label_query.shape[0]):
    y.append(np.argmax(label_query[i]))
y = np.asarray(y)

plt.plot(range(len(dataRes)), dataRes ,'r')
plt.plot(range(y.shape[0]), y , "g")
plt.show()