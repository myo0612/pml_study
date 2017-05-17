from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import plot_decision_regions as pdr
import wine_data
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

w = wine_data.WineDataSets()
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(w.X_train_std, w.y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, w.y_train)
pdr.plot_decision_regions(X_train_lda, w.y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
plt.show()

X_test_lda = lda.transform(w.X_test_std)
pdr.plot_decision_regions(X_test_lda, w.y_test, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')

plt.legend(loc = 'lower left')
plt.show()
