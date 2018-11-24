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
nb_classes = 4

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
	LABELS = ["スポーツ", "IT", "映画", "国際"]
	#テキストのTF-IDF値を求める
	data = tfidf.calc_text(text)
	#テキストのIF-IDF値を入力し、推論を行う
	pre = model.predict(np.array([data]))[0]
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
	text4 = "ドナルド・トランプは現在アメリカの大統領だけど、大丈夫かな？"
	text5 = "中国のAI産業の発展は目を見張るものがある。"
	text_list = [text1, text2, text3, text4, text5]
	for text in text_list:
		predict_genre(text)
