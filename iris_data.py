from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

class IrisDataSets(object):

	def __init__(self):
		iris = datasets.load_iris()
		self.X = iris.data[:, [2, 3]]
		self.y = iris.target
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)
		sc = StandardScaler()
		sc.fit(self.X_train)
		self.X_train_std = sc.transform(self.X_train)
		self.X_test_std = sc.transform(self.X_test)
		self.X_combined_std = np.vstack((self.X_train_std, self.X_test_std))
		self.y_combined = np.hstack((self.y_train, self.y_test))

	def useFit(self, method):
		method.fit(self.X_train_std, self.y_train)

	def drawGraph(self):
		plt.xlabel('petal length [Standardized]')
		plt.ylabel('petal width [Standardized]')
		plt.legend(loc = 'upper left')
		plt.show()

