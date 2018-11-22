import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json

#辞書のロード
tfidf.load_dic("data/pkl/dic.pkl")

#分類クラス数を指定
nb_classes = 3

#input = 17744
#input = np.array(input, dtype=float32)

#MLP(多層パーセプトロン)の定義
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(17744,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
#モデルのコンパイル
model.compile(
	loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])
#モデルに重みをロードする
model.load_weights('data/weights/genre-model.hdf5')

#引数で取得したテキストに対して、クラスを推論する関数
def check_genre(text):
	#クラス分類するラベルの指定
	LABELS = ["スポーツ", "IT", "映画"]
	#テキストのTF-IDF値を求める
	data = tfidf.calc_text(text)
	#テキストのIF-IDF値を入力し、推論を行う
	pre = model.predict(np.array([data]))[0]
	#with self.graph.as_default():
	#	pre = self.model.predict(np.array([data]))[0]

	#確率が最大となるインデックス値を返す
	n = pre.argmax()
	#推論結果のクラスを表示
	print(LABELS[n], "(", pre[n], ")")
	#推論結果のクラス、確率値、推論クラスのインデックスを返す
	return LABELS[n], float(pre[n]), int(n)

if __name__ == '__main__':
	#predict_using_mlp.pyをメインで実行するとテスト行う
	text1 = "昨日のサッカーの試合は面白すぎて興奮しました。"
	text2 = "スパイダーマンの新しいシリーズが待ち遠しい。"
	check_genre(text1)
	check_genre(text2)
