import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
	#マーカーとカラーマップの準備
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
			np.arange(x2_min, x2_max, resolution))
	#各特徴量を1次元配列に変換して予測を実行
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	'''
	print(Z)
	import csv
	f = open('test.csv', 'w')
	wcsv = csv.writer(f, lineterminator='n')
	wcsv.writerow(Z)
	'''

	#グリッドポイントの等高線のプロット
	plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#クラスごとにサンプルをプロット
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

	#テストサンプルを目立たせる
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c = '',
					alpha = 1.0, linewidths = 1, marker = 'o',
					s = 55, label = 'test set', edgecolors = 'black')