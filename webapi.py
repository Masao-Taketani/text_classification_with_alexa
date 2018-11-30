import json
import flask
from flask import request
import predict_using_mlp

#通信を行うポート番号の指定
PORT_NO = 8080
#Flaskインスタンスを生成
app = flask.Flask(__name__)

##Webアプリ用
#メインHTMLページを返す
@app.route('/', methods=['GET'])
def index():
    with open("index.html", "rb") as f:
        return f.read()

#テキストエリア内容から、カテゴリを予測し、結果をJSONで返す
@app.route('/api', methods=['GET'])
def api():
    # GETパラメータを取得
    q = request.args.get('q', '')
    if q == '':
      return '{"label": "テキストが空です", "percent":0}'
    print("q=", q)
    # テキストのジャンル判定を行い、予測したカテゴリとその確率を返す
    label, percent, _ = predict_using_mlp.predict_genre(q)
    #パーセント表示にする
    percent = round(percent * 100, 2)
    # 結果をJSONで出力
    return json.dumps({
      "label": label,
      "percent": percent
    })

##Alexa用
#リクエストとレスポンスの両方をJSONで通信する
@app.route('/', methods=['POST'])
def api_2():
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
