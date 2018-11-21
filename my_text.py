import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json


tfidf.load_dic("data/pkl/dic.pkl")

nb_classes = 3

#input = 17744
#input = np.array(input, dtype=float32)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(17744,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(
	loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])

model.load_weights('data/weights/genre-model.hdf5')

def check_genre(text):
	LABELS = ["スポーツ", "IT", "映画"]

	data = tfidf.calc_text(text)
	pre = model.predict(np.array([data]))[0]
	#with self.graph.as_default():
	#	pre = self.model.predict(np.array([data]))[0]
	n = pre.argmax()
	print(LABELS[n], "(", pre[n], ")")
	return LABELS[n], float(pre[n]), int(n)

if __name__ == '__main__':
	text1 = "昨日のサッカーの試合は面白すぎて興奮しました。"
	text2 = "スパイダーマンの新しいシリーズが待ち遠しい。"
	check_genre(text1)
	check_genre(text2)
