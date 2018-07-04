# _*_ coding: utf-8 _*_
from datetime import datetime
import pickle
import os
import h5py
import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, adam

start = datetime.now()

train = '../../output/cnn-keras/train.cnn'
test = '../../output/cnn-keras/test.cnn'

fo_tr = '../../output/cnn-keras/train.h5'
fo_te = '../../output/cnn-keras/test.h5'


def embeding(fi):
    embeding_dims = 15
    model = Sequential()
    model.add(Embedding(1358745, embeding_dims, input_length=15))
    model.compile('SGD', 'mse')
    x_train = []
    y_train = []
    fi = open(fi, 'r')
    for t, line in enumerate(fi, start=1):
        s = line.replace('\n', '').split(' ')
        input_array = []
        for x in s[1:]:
            input_array.append((int(x)))
        input_array = np.array(input_array).reshape(1, 15)
        # print(input_array)

        output_array = model.predict(input_array)
        x_train.append(output_array.reshape(15, embeding_dims))
        y_train.append(int(s[0]))

        if t % 200000 == 0:
            print(t)
            break

    x_train = np.expand_dims(x_train, 3)
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train


cache = True

if not cache:
    print('Embeding train set...')
    x_train, y_train = embeding(train)
    print('Embeding test set...')
    x_test, y_test = embeding(test)
else:
    # 如果有缓存，则从缓存中读取数据，否则构造数据
    if not os.path.exists(fo_tr):
        print('Embeding train data...')
        x_train, y_train = embeding(train)
        with h5py.File(fo_tr, 'w') as h:
            h.create_dataset('x_train', data=x_train)
            h.create_dataset('y_train', data=y_train)

    else:
        print('Loading train cache...')
        f = h5py.File(fo_tr, 'r')
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train'])
        f.close()
        print(x_train.shape)

    if not os.path.exists(fo_te):
        print('Embeding test data...')
        x_test, y_test = embeding(test)
        with h5py.File(fo_te, 'w') as h:
            h.create_dataset('x_test', data=x_test)
            h.create_dataset('y_test', data=y_test)
    else:
        print('Loading test cache...')
        f = h5py.File(fo_te, 'r')
        x_test = np.array(f['x_test'])
        y_test = np.array(f['y_test'])
        f.close()
        print(x_test.shape)

# set parameters:
max_features = 1358745
batch_size = 32

epochs = 100


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())
sgd = SGD(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())
print('training')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
          shuffle=True)

y_pred = model.predict(x_test, verbose=2)

auc = roc_auc_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred)
print('auc: {0}\tloglogg: {1}'.format(auc, logloss))


# 将预测结果写入文件
def pred_to_sub(y_pred):
    with open('../../output/cnn-keras/pred.csv', 'w') as fo:
        fo.write('id,prob\n')
        for t, prob in enumerate(y_pred, start=1):
            fo.write('{0},{1}\n'.format(t, float(prob)))

pred_to_sub(y_pred)

# print(model.get_weights())
print('耗时: {0}'.format(datetime.now()-start))

