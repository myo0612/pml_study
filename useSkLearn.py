from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
import iris_data as ir
import sys


args = sys.argv
def RunSkMethod(s = 'ppn'):
	isTree = False
	if s == 'ppn':
		method = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0, shuffle = True)
	elif s == 'lr':
		method = LogisticRegression(C = 100.0, random_state = 0)
	elif s == 'svc':
		method = SVC(kernel = 'linear', C=1.0, random_state = 0)
	elif s == 'svm':
		method = SVC(kernel = 'rbf', random_state = 0, gamma = float(args[2]), C = float(args[3]))
	elif s == 'tree':
		method = DTC(criterion='entropy', max_depth = 3, random_state = 0)
		isTree = True
	elif s == 'forest':
		method = RFC(criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs = 2)
	elif s == 'knn':
		method = KNC(n_neighbors = 5, p = 2, metric = 'minkowski') 
	elif s == 'pca':
		method = PCA(n_components = 2)
		return

	dd = ir.IrisDataSets()
	dd.useFit(method)
	pdr.plot_decision_regions(X = dd.X_combined_std, y = dd.y_combined, classifier = method, test_idx = range(105, 150))
	dd.drawGraph()

	if s == 'lr':
		print(method.predict_proba(dd.X_test_std[0,:].reshape(1, -1)))

	# after this function, execute following command on terminal
	# dot -Tpng tree.dot -o tree.png
	if isTree == True:
		export_graphviz(method, out_file = 'tree.dot', feature_names = ['petal length', 'petal width'])
		

# RunSkMethod()
