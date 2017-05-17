from sklearn.ensemble import RandomForestClassifier
import wine_data as wd
import numpy as np
import matplotlib.pyplot as plt 

w = wd.WineDataSets()
feat_labels = w.df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs = -1)
forest.fit(w.X_train, w.y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(w.X_train.shape[1]):
	print("%2d) %-*s %f", (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(w.X_train.shape[1]), importances[indices], color= 'lightblue', align = 'center')
plt.xticks(range(w.X_train.shape[1]), feat_labels[indices], rotation = 90) 
plt.xlim([-1, w.X_train.shape[1]])
plt.tight_layout()
plt.show()
