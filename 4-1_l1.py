from sklearn.linear_model import LogisticRegression
import wine_data as wd
import matplotlib.pyplot as plt
import numpy as np

w = wd.WineDataSets()
lr = LogisticRegression(penalty = 'l1', C = 0.1)
lr.fit(w.X_train_std, w.y_train)
print('training accuracy:', lr.score(w.X_train_std, w.y_train))
print('Test accuracy:', lr.score(w.X_test_std, w.y_test))
lr.intercept_
lr.coef_


fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4, 6):
	lr = LogisticRegression(penalty = 'l1', C = 10**np.int(c), random_state= 0)
	lr.fit(w.X_train_std, w.y_train)
	weights.append(lr.coef_[1])
	params.append(10**np.int(c))

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[:, column], label = w.df_wine.columns[column + 1], color = color)

plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight cofficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc = 'upper left', bbox_to_anchor = (1.38, 1.03), ncol = 1, fancybox = True)
plt.show()
