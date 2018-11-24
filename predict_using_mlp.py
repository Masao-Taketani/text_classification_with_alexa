import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json

#辞書のロード
tfidf.load_dic("data/pkl/dic.pkl")
#辞書の長さ-1をinput_sizeに設定
#(MLPの入力サイズを指定するために使用)
input_size = len(tfidf.word_dic) - 1

#分類クラス数を指定
nb_classes = 6

#MLP(多層パーセプトロン)の定義
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(input_size,)))
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
def predict_genre(text):
	#クラス分類するラベルの指定
	LABELS = ["スポーツ", "IT", "映画", "芸能", "ファッション", "国際"]
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
	text3 = "スマホはやっぱりアンドロイドだよね。"
	text4 = "大塚愛を最近ニュースで見たけど、旦那が浮気したのかな？"
	text5 = "最近寒くなってきたから、暖かくておしゃれなコートが欲しいな。"
	text6 = "ドナルド・トランプは現在アメリカの大統領だけど、大丈夫かな？"
	predict_genre(text1)
	predict_genre(text2)
	predict_genre(text3)
	predict_genre(text4)
	predict_genre(text5)
	predict_genre(text6)