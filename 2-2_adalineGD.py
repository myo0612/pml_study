import numpy as np
import matplotlib.pyplot as plt
import plot_decision_regions as pdr
import setData as sdt

class AdalineGD(object):
    """
    パラメータ
    ---------
    eta:float
        学習率(0~1.0)
    n_iter:int
        トレーニングデータのトレーニング回数

    属性
    --------
    w_:1次元配列
        適合後の重み
    errors_:リスト
        各エポックでの誤分類数
    """

    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        パラメータ
        --------
        X   :{配列のようなデータ構造}, shape = [n_samples, n_features]
             トレーニングデータ
             n_sampleはサンプルの個数、n_featureは特徴量の個数
        y   :配列のようなデータ構造, shape = [n_samples]
             目的変数
        戻り値
        --------
        self:object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
    
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def plotLearningCost():
    X, y = sdt.setSample_IrisSV()
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
    ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker = 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

def plotTraining():
    X, y = sdt.setSample_IrisSV()
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    ada = AdalineGD(n_iter = 15, eta = 0.01)
    ada.fit(X_std, y)
    pdr.plot_decision_regions(X_std, y, classifier = ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

# plotLearningCost()
plotTraining()
