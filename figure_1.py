import random

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.arange(-2.0, 3.2, 0.4)
std = 1.0
# y = x * np.sin(x) + np.random.normal(size=x.shape, scale=std)
noise = [np.random.normal(loc=1.5 * np.sin(2 * i), scale=std) for i in x]
y = np.square(x) + noise
x1 = np.arange(-2.0, 3.2, 0.01)

z = np.square(x1) + 2.5 * np.sin(2 * x1)

fig, axes = plt.subplots(1, 1)
axes.scatter(x, y, color='black', s=150)
axes.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# axes.tick_params(labelleft=False, labelbottom=False)

axes.set_xlim([-3.0, 3.5])
axes.set_ylim([-3.0, 11.0])
axes.spines['left'].set_linewidth(7)
axes.spines['bottom'].set_linewidth(7)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

fig.tight_layout()
fig.show()

fig, axes = plt.subplots(1, 1)
axes.scatter(x, y, color='black', s=150)
axes.plot(x1, z, color='red', linewidth=7)
axes.tick_params(left=False, right=False, labelleft=False,
                 labelbottom=False, bottom=False)
axes.set_xlim([-3.0, 3.5])
axes.set_ylim([-3.0, 11.0])
axes.spines['left'].set_linewidth(7)
axes.spines['bottom'].set_linewidth(7)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
fig.tight_layout()
fig.show()

fig, axes = plt.subplots(1, 1)
axes.scatter(x, y, color='black', s=150)
axes.plot(x1, np.square(x1), color='red', linewidth=7)
axes.tick_params(left=False, right=False, labelleft=False,
                 labelbottom=False, bottom=False)
axes.set_xlim([-3.0, 3.5])
axes.set_ylim([-3.0, 11.0])
axes.spines['left'].set_linewidth(7)
axes.spines['bottom'].set_linewidth(7)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
fig.tight_layout()
fig.show()

z2 = np.square(x1) + np.square(x1 - 1)
idx = np.argmin(z2)
fig, axes = plt.subplots(1, 1)
axes.plot(x1, z2, color='black', linewidth=7)
axes.scatter(x1[idx], z2[idx], color='red', s=1500, marker='*', zorder=2)
axes.tick_params(left=False, right=False, labelleft=False,
                 labelbottom=False, bottom=False)
axes.set_xlim([-3.0, 3.5])
axes.set_ylim([-3.0, 11.0])
axes.spines['left'].set_linewidth(7)
axes.spines['bottom'].set_linewidth(7)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
fig.tight_layout()
fig.show()

z3 = np.square(x1) + 2 * np.cos(2 * x1) - 2 * np.sin(2 * x1)
for i in range(x1.shape[0]):
    if i == 0:
        continue
    if z3[i - 1] > z3[i] and z3[i] < z3[i + 1]:
        idx = i
        break
fig, axes = plt.subplots(1, 1)
axes.plot(x1, z3, color='black', linewidth=7)
axes.scatter(x1[idx], z3[idx], color='red', s=1500, marker='*', zorder=2)
axes.tick_params(left=False, right=False, labelleft=False,
                 labelbottom=False, bottom=False)
axes.set_xlim([-3.0, 3.5])
axes.set_ylim([-3.0, 11.0])
axes.spines['left'].set_linewidth(7)
axes.spines['bottom'].set_linewidth(7)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
fig.tight_layout()
fig.show()
