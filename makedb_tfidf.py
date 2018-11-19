import os, glob, pickle
import tfidf

y = []
x = []

#引数に指定したファイルを読み込んでラベル付けを行う
def read_files(path, label):
	print("read_files=", path)
	#パス内にある全txtファイルを取得
	files = glob.glob(path + "/*.txt")
	#取得した全txtファイルを一つずループで見ていく
	for f in files:
		#LICENSE.txtはスキップする
		if os.path.basename(f) == 'LICENSE.txt':
			continue
		#txtファイルを
		tfidf.add_file(f)
		y.append(label)

read_files('data/text/sports-watch', 0)
read_files('data/text/it-life-hack', 1)
read_files('data/text/movie-enter', 2)

#TFIDFでベクトルに変換
x = tfidf.calc_files()

#保存
pickle.dump([y, x], open('data/pkl/genre.pickle', 'wb'))
tfidf.save_dic('data/dic/genre-tfidf.dic')
print('ok')