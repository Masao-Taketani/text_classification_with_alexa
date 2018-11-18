import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#分類するクラスのラベル数
nb_classes = 2

#データの読み込み
data = pickle.load(open("text/genre.pickle", "rb"))
#label
y = data[0]
#input
x = data[1]
#ラベルデータをone-hotベクトルに変換
y = keras.utils.np_utils.to_categorical(y, nb_classes)
x = np.array(x, dtype=np.float32)
in_size = x[0].shape[0]
print(in_size)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#MLP
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(
	loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])

hist = model.fit(
	x_train,
	y_train,
	batch_size=128,
	epochs=20,
	verbose=1,
	validation_data=(x_test, y_test)
	)

score = model.evaluate(x_test, y_test, verbose=1)
print("正解率＝", score[1], 'loss=', score[0])

model.save_weights('./text/genre-model.hdf5')
