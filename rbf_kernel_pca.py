from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
	"""RBFカーネルPCAの実装
	パラメータ
	--------
	X: {Numpy ndarray}, shape = [n_samples, n_features]
	gamma:float
		RBFカーネルのチューニングパラメータ
		グリッドサーチなどのパラ調アルゴリズムが必要
	n_components: int

	戻り値
	--------
	X_pc: {NumPy ndarray}, shape = [n_samples, k_features]

	"""
	# M*N次元のデータセットでペアごとの平方ユークリッド距離を計算
	sq_dists = pdist(X, 'sqeuclidean')
	
	#ペアごとの距離を正方行列に変換
	mat_sq_dists = squareform(sq_dists)

	#対象カーネル行列を計算
	K = exp(-gamma * mat_sq_dists)

	#カーネル行列を中心化
	N = K.shape[0]
	one_n = np.ones((N, N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	#中心化されたカーネル行列から固有対を取得
	#numpy.eighはそれらをソート順に返す
	eigvals, eigvecs = eigh(K)

	#上位k個の固有ベクトル(射影されたサンプル)を収集
	X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

	return X_pc

