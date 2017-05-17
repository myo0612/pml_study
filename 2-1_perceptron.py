import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_decision_regions as pdr
class Perceptron(object):
    """分類器
    parameter
    --------
    eta:float
        学習率
    n_iter:int
        トレーニングデータのトレーニング回数

    属性
    --------
    w_:1次元配列
        適合後の重み
    errors_:リスト
        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

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

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        #トレーニング回数分トレーニングデータを反復
        for _ in range(self.n_iter): 
            errors = 0
            for xi, target in zip(X, y):
                #重みw1, ... ,wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                #重みw0の更新
                self.w_[0] += update
                #重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)



def testTraining():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    # 1~100行目の目的変数を抽出
    y = df.iloc[0:100, 4].values
    # Iris-setosaを-1, Iris-virginicaを1に変換
    y = np.where(y == 'Iris-setosa', -1, 1)
    # 1~10行目の1,3列目を抽出
    X = df.iloc[0:100, [0, 2]].values
    #setosaのプロット
    plt.scatter(X[:50,0], X[:50,1], color='red', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker = 'x', label = 'versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc = 'upper left')
    plt.figure(0)

    ppn = Perceptron(eta =0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.figure(1)

    pdr.plot_decision_regions(X, y, classifier = ppn)
    plt.show()

testTraining()
