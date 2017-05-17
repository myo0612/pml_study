"""
【参考文献】
尤度について
	http://qiita.com/kenmatsu4/items/b28d1b3b3d291d0cc698
FuncAnimationについて
	http://qiita.com/AnchorBlues/items/3acd37331b12e844d259
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import numpy.random as rd
from matplotlib import animation as ani

num_frame = 30

min_x = -11
max_x = 21 
x = np.linspace(min_x, max_x, 201)

rd.seed(7)
data = rd.normal(10, 3, 10, )

def norm_dens(val, m, s):
	return (1 / np.sqrt(2 * np.pi * s**2))*np.exp(-0.5*(val - m)**2/s**2)

#mを変化させる
list_L = []
s = 3
mm = np.linspace(0, 20, 300)
for m in mm:
	list_L.append(np.prod([norm_dens(x_i, m, s) for x_i in data]))
plt.figure(0, figsize = (8, 5))
plt.xlim(min(mm), max(mm))
plt.plot(mm, (list_L))
plt.title("Likelihood curve")
plt.xlabel("mu")

#sを変化させる
list_L = []
m = 10
ss = np.linspace(0, 20, 300)
for s in ss:
	list_L.append(np.prod([norm_dens(x_i, m, s) for x_i in data]))
plt.figure(1, figsize = (8, 5))
plt.xlim(min(ss), max(ss))
plt.plot(ss, (list_L))
plt.title("Likelihood curve")
plt.xlabel("su")

#等高線
plt.figure(2, figsize = (8, 5))
mu = np.linspace(5, 15, 200)
s = np.linspace(0, 5, 200)
MU, S = np.meshgrid(mu, s)

Z = np.array([np.prod([norm_dens(x_i, a, b) for x_i in data]) for a, b in zip(MU.flatten(), S.flatten())])
plt.contour(MU, S, Z.reshape(MU.shape), cmap = cm.Blues)
plt.xlabel("mu")
plt.ylabel("s")

plt.show()

plt.show()




