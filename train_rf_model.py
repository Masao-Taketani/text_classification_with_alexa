import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np

data = pickle.load(open("data/pkl/genre.pickle", "rb"))
#label
y = data[0]
#TFIDF
x = data[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#ランダムフォレストで学習
model = RandomForestClassifier()
model.fit(x_train, y_train)

#評価して結果を出力
y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
rep = metrics.classification_report(y_test, y_pred)

print("正解率＝", acc)
print(rep)

#test = '今日は野球の試合がありますね。'
#y_pred = model.predict(test)
#print(y_pred)
