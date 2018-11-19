import MeCab
import pickle
import numpy as np

#tagger = MeCab.Tagger("-d /home/m-taketani/macab-ipadic-neologd")
#MeCabのTaggerインスタンスを生成
tagger = MeCab.Tagger()

#単語辞書
word_dic = {'_id': 0}
#全ての文書からそれぞれの単語の出現回数を記録
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

#指定の文書を'files'リストにidで格納する関数
def add_text(text):
	#指定の文書に分かち書きを行い、単語をidに
	#に変換後、変数idsで受け取る
	ids = words_to_ids(tokenize(text))
	#文書の単語をid化したものを'files'リストに格納
	#(filesリスト内には文書ごとに単語をid化したリストが入っている)
	files.append(ids)

#指定のパスにあるファイルを'utf-8'で開き、ファイル内の
#該当単語をidに変換後、'files'リストに格納する関数
def add_file(path):
	with open(path, "r", encoding="utf-8") as f:
		s = f.read()
		add_text(s)

#TF-IDFの計算を行う関数
def calc_files():
	#グローバル変数の'dt_dic(全ての文書からそれぞれの単語の出現回数を記録)'
	#辞書を使用
	global dt_dic
	#結果を返すためのリスト
	result = []
	#'files'リストの長さを変数'doc_count'に代入
	doc_count = len(files)
	#'dt_dic'辞書の初期化
	dt_dic = {}
	#'files'リストに入っているそれぞれの(分かち書きをしてidに変換済みの)文書
	#に対してループ処理
	for words in files:
		#辞書'used_word'の初期化
		used_word = {}
		#'word_dic['_id']'のidを抽出し、
		#そのidの数だけ要素をもち、全てその要素が0となる
		#numpy配列をdataに代入
		data = np.zeros(word_dic['_id'])
		#1つの文書内にあるそれぞれの単語に対してループ処理
		for id in words:
			#dataのid番目を1プラスする
			data[id] += 1
			#used_wordのid番目を1に設定する
			used_word[id] = 1
		#単語tが使われていればdt_dicを加算
		for id in used_word:
			#'dt_dic'内に指定のidがない場合、'dt_dic'の
			#keyにidを追加し、valueを0に初期設定する。
			if not(id in dt_dic):
				dt_dic[id] = 0
			#dt_dicのkeyがidであるvalueに対して1を加算(出現回数を+1)
			dt_dic[id] += 1
		#1つの文書の単語の出現回数を文書内に出てきた総単語数で割り、
		#それぞれの単語の出現割合を求める(TF値)
		data = data / len(words)
		#それぞれの単語の出現割合を'result'リストに追加
		result.append(data)

	##TF-IDFを計算を行う
	#文書ごとにループ処理
	for i, doc in enumerate(result):
		#それぞれの文書内の単語ごとにループ処理
		for id, v in enumerate(doc):
			#idfの計算
			idf = np.log(doc_count / dt_dic[id]) + 1
			#TFxIDFの値を計算し、計算結果と1を比べて小さい方の値を
			#'doc[id]'に代入(＝最大を1とする)
			doc[id] = min([doc[id] * idf, 1.0])
		#i番目の文書ごとにそれぞれの単語に対して算出したTF-IDF値
		#を'result'リストのi番目に代入
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
