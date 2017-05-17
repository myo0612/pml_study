import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_decision_regions as pdr
import setData as sdt
from numpy.random import seed

class AdalineSGD(object):
	"""
	パラメータ
	--------
	eta:float
		学習率(0~0.1)
	n_iter:int
		トレーニングデータのトレーニング回数
	
	属性
	---------
	w_:1次元配列
		適合後の重み
	errors_:リスト
		各エポックでの誤分類数
	shuffle:bool(デフォルト:True)
		循環を回避するために各エポックでトレーニングデータをシャッフル
	random_state:int(デフォルト:None)
		シャッフルに使用するrandom_stateを設定し、重みを初期化
	"""

	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		if random_state:
			seed(random_state)
	
	def fit(self, X, y):
		"""トレーニングデータに適合させる
		パラメータ
		---------
		X:（配列のようなデータ構造), shape = [n_samples, n_features]
		  トレーニングデータ
		  n_sampleはサンプルの個数、ndeatureは特徴量の個数
		y:配列のようなデータ構造, shape = [n_samples]
		  目的変数

		戻り値
		----------
		self:object
		"""
		self._initialize_weights(X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
			cost = []

			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))

			avg_cost = sum(cost) / len(y)
			self.cost_.append(avg_cost)

		return self

	def partial_fit(self, X, y):
		"""重みを初期化することなくトレーニングデータに適合させる"""
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		#目的変数yの要素数が2以上の場合は
		#各サンプルの特徴量xiと目的変数targetで重みを更新
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)
		#目的変数yの要素数が1の場合は
		#サンプル全体の特徴量Xと目的変数yで重みを更新
		else:
			self._update_weights(X, y)

		return self

	def _shuffle(self, X, y):
		r = np.random.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):
		self.w_ = np.zeros(1 + m)
		self.w_initialized = True

	def _update_weights(self, xi, target):
		"""ADALINEの学習規則を用いて重みを更新"""
		output = self.net_input(xi)
		error = (target - output)
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error**2
		return cost

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)


def plotTraining():
	X, y = sdt.setSample_IrisSV()
	X_std = np.copy(X)
	X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
	ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
	ada.fit(X_std, y)
	pdr.plot_decision_regions(X_std, y, classifier = ada)
	plt.title('Adaline - Stochastic Gradient Descent')
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.legend(loc = 'upper left')
	plt.tight_layout()
	plt.figure(0)
	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
	plt.xlabel('Epochs')
	plt.ylabel('Average Cost')
	plt.figure(1)
	plt.show()

plotTraining()
