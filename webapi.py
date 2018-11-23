import json
import flask
from flask import request
import predict_using_mlp

#通信を行うポート番号の指定
PORT_NO = 8080
#Flaskインスタンスを生成
app = flask.Flask(__name__)
#デコレーターを設定し、関数apiを定義
@app.route('/', methods=['POST'])
def api():
	#'Content-Type'が'application/json'以外の場合
	if request.headers['Content-Type'] != 'application/json':
		print('Content-Type:',request.headers['Content-Type'])
		#400番エラーを返す
		return flask.jsonify(res='error'), 400
	else:
		#JSONの値を'data'にて取得
		data = request.get_json()
		#'data'内にあるキー値が'text'のバリュー値を表示
		print(data['text'])
		#取得したテキストに対してクラスを推論し、ラベルのみを受け取る
		label, _ , _ = predict_using_mlp.predict_genre(data['text'])
		#ラベルをキー値:"label",バリュー値:変数'label'の形で返す(JSON形式)
		return json.dumps({"label": label})

#サーバの起動
if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0', port=PORT_NO, threaded=False)
