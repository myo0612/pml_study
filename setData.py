import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def setSample_IrisSV():
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
    return (X, y)
