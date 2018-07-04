# _*_ coding: utf-8 _*_
from datetime import datetime
import pickle
import os
import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, Conv1D, MaxPooling2D, Flatten, Input


start = datetime.now()

train = '../../output/cnn-keras/train.cnn'
test = '../../output/cnn-keras/test.cnn'

x_train = []
y_train = []
fi = open(train, 'r')
for t, line in enumerate(fi, start=1):
    s = line.replace('\n', '').split(' ')
    input_array = []
    for x in s[1:]:
        input_array.append((int(x)))
    input_array = np.array(input_array).reshape(1, 16)
    x_train.append(input_array)
    y_train.append(int(s[0]))
    if t % 100000 == 0:
        print(t)
        break

print(np.array(x_train).shape)

model = Sequential()
model.add(Dense(16, input_shape=(1, 16)))
model.add(Embedding(1358752, 16, input_length=16, trainable=False))

model.add(Conv2D(32, (4, 4), padding='same', activation='relu'))
model.add(Conv2D(32, (4, 4), padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('training')
model.fit(x_train, y_train,
          batch_size=1000,
          epochs=10)
