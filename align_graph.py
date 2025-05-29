import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d # smooth 1d data using gaussian filter
from sklearn.decomposition import PCA # PCA reduces high dimensions
from sklearn.cross_decomposition import CCA #aligning
from mpl_toolkits.mplot3d import Axes3D #3d plotting in mpl

def align_cca(X, Y, n_components = None):
    # computes CCA allignment between X and Y
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])
    cca = CCA(n_components = n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    return X_c, Y_c, cca

# test align_cca
# from sklearn.datasets import load_wine

# wine_data = load_wine()
# X_full = wine_data.data

# X = X_full[:, :6] # first 6
# Y = X_full[:, 6:] # 6 and forward
# X_c, Y_c, cca_test = align_cca(X, Y)
# print(X_c.shape)
# print(Y_c.shape)

# for i in range(X_c.shape[1]): # finds correlation constant
#     r = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
#     print(f"Component {i+1} correlation: {r:.2f}")


# CHARACTERISICS OF CCA WITH GEOMETRIC DATA

T = 200
t = np.linspace(0, 2*np.pi, T) # T spaced 0 to 2pi (aka one rev)
z = t/(2*np.pi) # z coordinate [0, 1]

#example tragectory
A = np.row_stack((np.cos(t), np.sin(t), z)).T

#linear insensitive
AQR = np.row_stack((A[:,0], A[:,1], 0.05*A[:,2])).T

#shape sensitive
r = 0.5 + z #explanding radius
B = np.row_stack((r*np.cos(t), r*np.sin(t), z)).T

#aligning noises
np.random.seed(1)
N = np.random.randn(T, 3) *0.5

#Aligning noise tragectory
AN = A + N*0.5

#librarys
datasets = [AQR, B, N, AN]
labels = ['AQR', 'B', 'N', 'AN']
aligned_cca = []
correlation_coefficients = []

pca = PCA(n_components = 3)
X = pca.fit_transform(A) # PCA of A

for data in datasets:
    Y = pca.fit_transform(data)
    X_c, Y_c, __ = align_cca(X, Y)
    aligned_cca.append(Y_c)
    #corelation coefficient
    r_vals = [np.corrcoef((X_c[:, i], Y_c[:, i]))[0, 1] for i in range(3)]
    correlation_coefficients.append(np.mean(r_vals))


#graphing

fig = plt.figure(figsize = (13,6))

ax = fig.add_subplot(2,5,1, projection = '3d')
ax.plot(A.T[1], A.T[0], A.T[2], label = "A")
ax.legend(loc = 'upper right')

for i, (raw, lbl) in enumerate(zip(datasets, labels)):
    ax = fig.add_subplot(2, 5, i+2, projection = '3d')
    ax.plot(raw.T[1], raw.T[0], raw.T[2], label = lbl)
    ax.set_zlim(0,1)
    ax.legend(loc = 'upper right')

for i, (Yc, lbl, cc) in enumerate(zip(aligned_cca, labels, correlation_coefficients)):
    ax = fig.add_subplot(2, 5, i+7, projection = '3d')
    ax.plot(X_c.T[1], X_c.T[0], X_c.T[2], label = 'A')
    ax.plot(Yc.T[1], Yc.T[0], Yc.T[2], label = f"{lbl} to A", alpha = 0.75)
    ax.set_title(f"μ CC: {cc:.2f}")
    ax.legend(loc = 'upper right')

fig.suptitle("Characteristics of CCA with Geometric Data", fontsize=16)

plt.show()


# #AQR
# fig = plt.figure(figsize = (4,4))
# axes = fig.add_subplot(111, projection = "3d")
# axes.plot(b.T[1], b.T[0], b.T[2], label = "AQR") #rotated
# axes.set_zlim(0,1))
# axes.legend()
# plt.show()

# print(AQR)
# print(A)

# #A
# fig = plt.figure(figsize = (4,4))
# axes = fig.add_subplot(111, projection = "3d")
# axes.plot(A.T[1], A.T[0], A.T[2], label = "A") #rotated
# axes.legend()
# plt.show()

