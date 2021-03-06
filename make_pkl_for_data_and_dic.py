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
read_files('data/text/international', 3)

#TFIDFでそれぞれの単語をベクトルに変換(辞書更新も行う)
x = tfidf.calc_files()

#pickleファイルを格納するフォルダディレクトリ
dir_for_pkls = 'data/pkl/'
#上記のディレクリを再帰的に作成する
os.makedirs(dir_for_pkls, exist_ok=True)
#データ格納用pickleファイル
pkl_for_data = "data.pkl"
#入力値とラベルを追加したリストを上記のpickleファイルに格納(バイナリ形式)
pickle.dump([y, x], open(dir_for_pkls + pkl_for_data, 'wb'))
if os.path.exists(dir_for_pkls + pkl_for_data):
	print('データpickleファイルの保存完了')
else:
	print('データpickleファイルの保存に失敗しました')

#辞書格納用pickleファイル
pkl_for_dic = 'dic.pkl'
#[word_dic, dt_dic, files]をpickleファイルに格納(バイナリ形式)
tfidf.save_dic(dir_for_pkls + pkl_for_dic)
if os.path.exists(dir_for_pkls + pkl_for_dic):
	print('辞書pickleファイルの保存完了')
else:
	print('辞書pickleファイルの保存に失敗しました')
