import os, glob, pickle
import tfidf

#ラベル格納用リスト
y = []
#それぞれの単語に対してTF-IDFでベクトル化した入力値を格納するリスト
x = []

#引数に指定したファイルを読み込んで指定したラベル付けを行う関数
def read_files(path, label):
	print("read_files=", path)
	#パス内にある全txtファイルを取得
	files = glob.glob(path + "/*.txt")
	#取得した全txtファイルを一つずループで見ていく
	for f in files:
		#LICENSE.txtはスキップする
		if os.path.basename(f) == 'LICENSE.txt':
			continue
		#txtファイルを学習文書リスト'files'に格納する
		tfidf.add_file(f)
		#格納したテキストに対応するラベルを変数'y'に追加
		y.append(label)

#カテゴリごとに分かれたディレクトリ内にあるデータに対してラベル付けを行う
read_files('data/text/sports-watch', 0)
read_files('data/text/it-life-hack', 1)
read_files('data/text/movie-enter', 2)

#TFIDFでそれぞれの単語をベクトルに変換(辞書更新も行う)
x = tfidf.calc_files()

#入力値とラベルを追加したリストをpickleファイルに格納(バイナリ形式)
path_of_pickle_for_input_output = 'data/pkl/data.pkl'
pickle.dump([y, x], open(path_of_pickle_for_input_output, 'wb'))
if is os.path.exists(path_of_pickle_for_input_output):
	print('pickleファイルの保存完了')
else:
	print('pickleファイルの保存に失敗しました')

#[word_dic, dt_dic, files]を格納(バイナリ形式)
path_of_dic = 'data/pkl/dic.pkl'
tfidf.save_dic(path_of_dic)
if os.path.exists(path_of_dic):
	print('辞書の保存完了')
else:
	print('辞書の保存に失敗しました')