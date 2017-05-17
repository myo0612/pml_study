from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
	"""
	逐次後退選択
	"""

	def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = 0.25, random_state = 1):
		self.scoring = scoring
		self.estimator = estimator
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
		# すべての特徴量の個数と列インデックス	
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		# すべての特徴量を用いてスコアを算出
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score]
		# 指定した特徴量の個数になるまで処理を反復
		while dim > self.k_features:
			scores, subsets = [], []
			# 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
			for p in combinations(self.indices_, r = dim - 1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)

			# 最良のスコアのインデックスを抽出
			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1

			self.scores_.append(scores[best])

		self.k_score_ = self.scores_[-1]

		return self

	def transform(self, X):
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		# 指定された列番号indicesの特徴y楼を抽出してモデルに適合
		self.estimator.fit(X_train[:, indices], y_train)
		# テストデータを用いてクラスラベルを予測
		y_pred = self.estimator.predict(X_test[:, indices])
		# 心のクラスラベルと予測値を用いてスコア算出
		score = self.scoring(y_test, y_pred)
		return score

