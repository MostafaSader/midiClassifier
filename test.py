import numpy as np
from keras.models import load_model
import funcs

model = load_model('./model.h5')
print("get test data")

data_test, label_test = funcs.get_test_data()
f = funcs.extract(data_test)

output_data = []
for item in label_test:
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

res = model.evaluate(f, output_data)
print(res)
print(model.metrics_names)

res = model.predict(f)

for i, item in enumerate(res):
    idxMax = np.argmax(item)
    predict = 0

    if idxMax == 0:
        predict = 3
    elif idxMax == 1:
        predict = 5
    elif idxMax == 2:
        predict = 6
    elif idxMax == 3:
        predict = 7
    elif idxMax == 4:
        predict = 9
    print("predict " + str(predict) + " real " + str(label_test[i]))

# res = ( res * 100).astype('int16')
# model.save('./test128.h5')
