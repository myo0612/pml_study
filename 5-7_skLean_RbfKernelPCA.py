from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import rbf_kernel_pca as RKP

X, y = make_moons(n_samples = 100, random_state = 123)

skKpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_skernpca = skKpca.fit_transform(X)
plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color = 'r', marker = '^', alpha = 0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color = 'b', marker = 'o', alpha = 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
