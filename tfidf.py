import MeCab
import pickle
import numpy as np

#tagger = MeCab.Tagger("-d /home/m-taketani/macab-ipadic-neologd")
#MeCabのTaggerインスタンスを生成
tagger = MeCab.Tagger()

#単語辞書
word_dic = {'_id': 0}
#文書全体で単語の出現回数
dt_dic = {}
#全文書をIDで保存するためのリスト
files = []

##分かち書きを行う関数
def tokenize(text):
	#結果を返すためのリスト
	result = []
	#taggerインスタンスによる解析結果をword_sに格納
	word_s = tagger.parse(text)
	#解析結果を改行ごとに分割後、ループ処理
	for n in word_s.split("\n"):
		#文字が'EOS'、または空の場合、スキップ
		if n == 'EOS' or n == '':
			continue
		#解析結果をタブで分割し、2番目の内容をpに代入
		p = n.split("\t")[1].split(",")
		#p[0]は品詞、p[1]は品詞細分類1、p[6]は原形
		h, h2, org = (p[0], p[1], p[6])
		#品詞が'名詞', '動詞', でないときスキップ
		if not (h in ['名詞', '動詞', '形容詞']):
			continue
		#品詞が'名詞'かつ品詞細分類1が'数'のときスキップ
		if h == '名詞' and h2 == '数':
			continue
		#上にある3つのif文でスキップされなかった原形を'result'に追加
		result.append(org)
	return result

##単語をIDに変換する関数(引数のwordsには単語のリストを指定)
def words_to_ids(words, auto_add = True):
	#結果を返すためのリスト
	result = []
	#wordsリストのそれぞれの単語に対してループ
	for w in words:
		#単語が既に単語辞書の中にある場合
		#その単語に該当するidを'result'リストに追加
		if w in word_dic:
			result.append(word_dic[w])
		#単語が単語辞書に存在しない場合は'auto_add'が
		#Trueのときに辞書に単語を追加を行い、
		#その単語のidを'result'リストに追加
		elif auto_add:
			id = word_dic[w] = word_dic['_id']
			word_dic['_id'] += 1
			result.append(id)
	return result

#指定のテキストを'files'リストにidで格納する関数
def add_text(text):
	#指定のテキストに分かち書きを行い、単語をidに
	#に変換後、変数idsで受け取る
	ids = words_to_ids(tokenize(text))
	#テキストをid化したものを'files'リストに格納
	files.append(ids)

#指定のパスにあるファイルを'utf-8'で開き、ファイル内の
#該当単語をidに変換後、'files'リストに格納する関数
def add_file(path):
	with open(path, "r", encoding="utf-8") as f:
		s = f.read()
		add_text(s)


def calc_files():
	global dt_dic
	result = []
	doc_count = len(files)
	dt_dic = {}
	#単語の出現頻度を数える
	for words in files:
		used_word = {}
		data = np.zeros(word_dic['_id'])
		for id in words:
			data[id] += 1
			used_word[id] = 1
		#単語tが使われていればdt_dicを加算
		for id in used_word:
			if not(id in dt_dic):
				dt_dic[id] = 0
			dt_dic[id] = 1
		#出現回数を割合に直す
		data = data / len(words)
		result.append(data)
	#IFIDFを計算
	for i, doc in enumerate(result):
		for id, v in enumerate(doc):
			idf = np.log(doc_count / dt_dic[id]) + 1
			doc[id] = min([doc[id] * idf, 1.0])
		result[i] = doc
	return result

def save_dic(fname):
	pickle.dump(
		[word_dic, dt_dic, files],
		open(fname, "wb"))

def load_dic(fname):
	global word_dic, dt_dic, files
	n = pickle.load(open(fname, 'rb'))
	word_dic, dt_dic, files = n

def calc_text(text):
	data = np.zeros(word_dic['_id'])
	words = words_to_ids(tokenize(text), False)
	for w in words:
		data[w] += 1
	data = data / len(words)
	for id, v in enumerate(data):
		idf = np.log(len(files) / dt_dic[id]) + 1
		data[id] = min([data[id] * idf, 1.0])
	return data

if __name__ == '__main__':
	add_text('雨')
	add_text('今日は、雨が降った。')
	add_text('今日は暑い日だったけど雨が降った。')
	add_text('今日も雨だ。でも日曜だ。')
	print(calc_files())
	print(word_dic)
