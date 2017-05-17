import wine_data
import plot_decision_regions as pdr
import numpy as np
import matplotlib.pyplot as plt

w = wine_data.WineDataSets()
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(w.X_train_std[w.y_train == label], axis = 0))
    print('MV %s: %s\n' %(label, mean_vecs[label - 1]))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    """
    スケーリングなし
    class_scatter = np.zeros((d, d))
    for row in w.X_train_std[w.y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
    """
#スケーリングあり
    class_scatter = np.cov(w.X_train_std[w.y_train == label].T)
    S_W += class_scatter


print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
#クラスラベルの個数を出力->分布が一様でないことがわかる
print('Class label distribution: %s' % np.bincount(w.y_train)[1:])

mean_overall = np.mean(w.X_train_std, axis = 0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = w.X_train[w.y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse = True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha = 0.5, align = 'center', label = 'individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label = 'cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc = 'best')
plt.show()

W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', W)

X_train_lda = w.X_train_std.dot(W)
colors  = ['r', 'b', 'g']
markers  = ['s', 'x', 'o']
for l, c, m in zip(np.unique(w.y_train), colors, markers):
    plt.scatter(X_train_lda[w.y_train == l, 0] * (-1), X_train_lda[w.y_train == l, 1] * (-1), c = c, label = l, marker = m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower right')
plt.savefig("lba_compress.png")
plt.show()

