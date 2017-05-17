import numpy as np
import matplotlib.pyplot as plt
import plot_decision_regions as pdr
from sklearn.svm import SVC

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c ='b', marker = 'x', label = '1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c ='r', marker = 's', label = '-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc = 'upper left')

svm = SVC(kernel='rbf', random_state = 0, gamma = 0.10, C=10.0)
svm.fit(X_xor, y_xor)
pdr.plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.show()
