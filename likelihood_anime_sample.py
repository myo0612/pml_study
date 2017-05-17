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

def animate(nframe):
	global num_frame
	plt.clf()

	# m = nframe/float(num_frame) * 15
	# s = 3
	m = 10
	s = nframe / float(num_frame) * 5
	y = norm_dens(x, m, s)

	L = np.prod([norm_dens(x_i, m, s) for x_i in data])
	l = np.log(L)

	plt.xlim(min_x, 16)
	plt.ylim(-0.01, max(y)*1.1)
	plt.plot(x, y)
	plt.scatter(data, np.zeros_like(data), c = "r", s = 50)
	for d in data:
		plt.plot([d, d], [0, norm_dens(d, m, s)], "k--", lw = 1)

	plt.title("mu:{0}, Likelihood:{1:5f}, log Likelihood:{2:5f}".format(s, L, l))

fig = plt.figure(figsize = (10, 7))
anim = ani.FuncAnimation(fig, animate, frames = int(num_frame))
# blit=Trueのオプションは削除して実行した。
# anim = ani.FuncAnimation(fig, animate, frames = int(num_frame), blit = True)
anim.save('Likelihood.gif', writer = 'imagemagick', fps = 1, dpi = 64)
