# http://qiita.com/kenmatsu4/items/b28d1b3b3d291d0cc698
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import numpy.random as rd

m = 5
s = 4
min_x = m - 4 * s
max_x = m + 4 * s

def norm_dens(val):
	return (1 / np.sqrt(2 * np.pi * s**2))*np.exp(-0.5*(val - m)**2/s**2)

x = np.linspace(min_x, max_x, 201)
y = norm_dens(x)

rd.seed(7)
data = rd.normal(10, 3, 10, )

L = np.prod([norm_dens(x_i) for x_i in data])
l = np.log(L)

plt.figure(figsize = (8, 5))
plt.xlim(min_x, 16)
plt.ylim(-0.01, max(y)*1.1)

plt.plot(x, y)

plt.scatter(data, np.zeros_like(data), c = "r", s = 50)
for d in data:
	plt.plot([d, d], [0, norm_dens(d)], "k--", lw = 1)

plt.title("Likelihood:{0:5f}, log Likelihood:{1:5f}".format(L, l))

plt.show()
