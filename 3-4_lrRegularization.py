from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
import iris_data as ir
import useSkLearn as sk

dd = ir.IrisDataSets()
weights, params = [], []
for c in np.arange(-5, 5):
	lr = LogisticRegression(C = 10**np.int(c), random_state = 0)
	lr.fit(dd.X_train_std, dd.y_train)
	weights.append(lr.coef_[1])
	params.append(10**np.int(c))

weights = np.array(weights)
plt.plot(params, weights[:, 0], label = 'petal length')
plt.plot(params, weights[:, 1], linestyle = '--', label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('c')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

