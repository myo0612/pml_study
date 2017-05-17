import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wine_data as wd
from sklearn.linear_model import LogisticRegression
import plot_decision_regions as pdr

pca = PCA(n_components = 2)
lr = LogisticRegression()
w = wd.WineDataSets()
X_train_pca = pca.fit_transform(w.X_train_std)
X_test_pca = pca.transform(w.X_test_std)
lr.fit(X_train_pca, w.y_train)
pdr.plot_decision_regions(X_train_pca, w.y_train, classifier = lr)
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.legend(loc = 'lower left')
plt.show()

pdr.plot_decision_regions(X_test_pca, w.y_test, classifier = lr)
plt.xlabel('PC1')
plt.xlabel('PC2')
plt.legend(loc = 'lower left')
plt.show()

