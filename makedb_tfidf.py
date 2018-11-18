import os, glob, pickle
import tfidf

y = []
x = []

def read_files(path, label):
	print("read_files=", path)
	files = glob.glob(path + "/*.txt")
	for f in files:
		if os.path.basename(f) == 'LICENSE.txt':
			continue
		tfidf.add_file(f)
		y.append(label)

read_files('text/sports-watch', 0)
read_files('text/it-life-hack', 1)

#TFIDFベクトルに変換
x = tfidf.calc_files()

#保存
pickle.dump([y, x], open('text/genre.pickle', 'wb'))
tfidf.save_dic('text/genre-tfidf.dic')
print('ok')