from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
import iris_data as ir

def sklearn_perceptron():
	iris = datasets.load_iris()
	X = iris.data[:, [2, 3]]
	y = iris.target
	print("Class labels:", np.unique(y))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0, shuffle = True)
	ppn.fit(X_train_std, y_train)

	y_pred = ppn.predict(X_test_std)
	print('Misclassified samples: %d' % (y_test != y_pred).sum())	
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

	X_combined_std = np.vstack((X_train_std, X_test_std))
	y_combined = np.hstack((y_train, y_test))
	pdr.plot_decision_regions(X = X_combined_std, y = y_combined, classifier = ppn, test_idx = range(105,150))
	plt.xlabel('petal length [standardized]')
	plt.ylabel('petal width [standardized]')
	plt.legend(loc = 'upper left')
	plt.show()

def sklean_p():
	dd = ir.IrisDataSets()
	ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0, shuffle = True)
	dd.useFit(ppn)
	pdr.plot_decision_regions(X = dd.X_combined_std, y = dd.y_combined, classifier = ppn, test_idx = range(105, 150))
	dd.drawGraph()

#sklearn_perceptron()
sklean_p()	