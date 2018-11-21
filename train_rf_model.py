import pickle, tfidf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np

data = pickle.load(open("data/pkl/data.pkl", "rb"))
#ラベルをyに代入
y = data[0]
#IF-IDF計算された入力をxに代入
x = data[1]

#学習用データとテスト用データに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#ランダムフォレスト(RF)を行うクラスをインスタンス化
model = RandomForestClassifier()
#RFで学習
model.fit(x_train, y_train)

#テストデータに対して推論を行う
y_pred = model.predict(x_test)
#推論結果と正解ラベルの精度を求める
acc = metrics.accuracy_score(y_test, y_pred)
#推論結果に対して'適合率(precision)'、'再現率(recall)'、
#'F値(f1-score=適合率と再現率の調和平均)'を求める
#('support' is the number of occurrences of each class in y_true)
rep = metrics.classification_report(y_test, y_pred)

#精度の表示
print("精度＝", acc)
#classification_reportの結果を表示
print(rep)

#入力したテキストに対して推論を結果をテスト
LABELS = ["スポーツ", "IT", "映画"]
test = '今日は野球の試合がありますね。'
preprocessed_input = tfidf.calc_text(test)
y_pred = model.predict(np.array([preprocessed_input]))[0]
mle_index = y_pred.argmax()
print(LABELS[mle_index])