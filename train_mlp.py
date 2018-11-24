import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#分類するクラスのラベル数
nb_classes = 4

#データの読み込み
data = pickle.load(open("data/pkl/data.pkl", "rb"))
#正解データを変数yに代入
y = data[0]
#入力データを変数xに代入
x = data[1]
#正解データをone-hotベクトルに変換
y = keras.utils.np_utils.to_categorical(y, nb_classes)
#入力データをnumpy配列のfloat32型に変換
x = np.array(x, dtype=np.float32)
#入力サイズの設定
in_size = x[0].shape[0]
#学習用データと検証用データに指定の割合で分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

##MLP(多層パーセプトロンの実装)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
#作成したモデルのコンパイル
model.compile(
	loss='categorical_crossentropy',
	optimizer=RMSprop(),
	metrics=['accuracy'])
#学習を行う
hist = model.fit(
	x_train,
	y_train,
	batch_size=128,
	epochs=20,
	verbose=1,
	validation_data=(x_test, y_test)
	)

#テストデータに対して推論を行う
y_pred = model.predict(x_test)
#それぞれの予測値で最大の値を持つインデックスを格納
y_pred_argmaxes = np.argmax(y_pred, axis=1)
#それぞれの正解値で最大の値を持つインデックスを格納
y_test_argmaxes = np.argmax(y_test, axis=1)
#精度計算
acc = metrics.accuracy_score(y_test_argmaxes, y_pred_argmaxes)
#推論結果に対して'適合率(precision)'、'再現率(recall)'、
#'F値(f1-score=適合率と再現率の調和平均)'を求める
#('support' is the number of occurrences of each class in y_true)
rep = metrics.classification_report(y_test_argmaxes, y_pred_argmaxes)

print("精度：", acc)
print(rep)

#学習モデルを保存するディレクトリ
dir_for_weights = "data/weights/"
#ディレクトリの作成
os.makedirs(dir_for_weights, exist_ok=True)
#学習から得たモデルの重みを保存
model.save_weights('data/weights/genre-model.hdf5')
if os.path.exists('data/weights/genre-model.hdf5'):
	print('モデルの重み保存完了')
