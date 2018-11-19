import json
import flask
from flask import request
import my_text

PORT_NO = 8080
#boot HTTP server
app = flask.Flask(__name__)

@app.route('/', methods=['POST'])
def api():
	#URL
	if request.headers['Content-Type'] != 'application/json':
		print('test2')
		print(request.headers['Content-Type'])
		return flask.jsonify(res='error'), 400
	data = request.get_json()
	print(data['text'])
	
	label, percent, no = my_text.check_genre(data['text'])
	return json.dumps({
		"label": label
		})

if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0', port=PORT_NO, threaded=False)
