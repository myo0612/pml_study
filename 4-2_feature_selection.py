import sbs as sbsFunc 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import wine_data as wd
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = sbsFunc.SBS(knn, k_features=1)
w = wd.WineDataSets()
sbs.fit(w.X_train_std, w.y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.xlim([0, 14])
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(w.df_wine.columns[1:][k5])

knn.fit(w.X_train_std, w.y_train)
print('Training accuracy', knn.score(w.X_train_std, w.y_train))
print('Test accuracy', knn.score(w.X_test_std, w.y_test))

knn.fit(w.X_train_std[:, k5], w.y_train)
print('After feature-selection')
print('Training accuracy', knn.score(w.X_train_std[:, k5], w.y_train))
print('Test accuracy', knn.score(w.X_test_std[:, k5], w.y_test))
