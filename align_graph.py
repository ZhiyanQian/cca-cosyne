import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d # smooth 1d data using gaussian filter
from sklearn.decomposition import PCA # PCA reduces high dimensions
from my_functions import CCA
from mpl_toolkits.mplot3d import Axes3D #3d plotting in mpl

# def align_cca(X, Y, n_components = None):
#     # computes CCA allignment between X and Y
#    if n_components is None:
#         n_components = min(X.shape[1], Y.shape[1])
#     cca = CCA(n_components = n_components)
#     X_c, Y_c = cca.fit_transform(X, Y)
#     return X_c, Y_c, cca

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
np.random.seed(1)
M = np.random.randn(3, 3)
Q_rand, R_base = np.linalg.qr(M)
R_compress = np.diag([1.0, 1.0, 0.05])
#Rotate (Q) and scale (R) -> QR decomposition
AQR = (A @ Q_rand) @ (R_base @ R_compress)


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
    S, Y_aligned_to_X = CCA(X, Y, align='B2A')
    Q_A, R_A = np.linalg.qr(X)        # X = Q_A @ R_A; x coordinates
    Q_B, R_B = np.linalg.qr(Y)        # Y = Q_B @ R_B; y coordinates
    U, _, V_T = np.linalg.svd(Q_A.T @ Q_B)
    X_c = (Q_A @ U[:, :3]) @ R_A      
    Y_c = Y_aligned_to_X[:, :3]       
    aligned_cca.append(Y_c)
    #corelation coefficient
    correlation_coefficients.append(np.mean(S[:3]))


#graphing -- objected based graphing code from chatgpt

fig, axes = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(13, 6),
    subplot_kw={'projection': '3d'}
)

# top‐left: original A (bas
axes[0, 0].plot(A[:, 1], A[:, 0], A[:, 2], color='tab:green', lw=2)
axes[0, 0].set_title("A (base)")
axes[0, 0].set_zlim(0, 1)
 
# top‐row raw transformations (columns 1..4 in the top row)
for idx, (raw, lbl) in enumerate(zip(datasets, labels)):
    ax = axes[0, idx+1]
    ax.plot(raw[:, 1], raw[:, 0], raw[:, 2], color='darkgray', lw=1)
    ax.set_title(lbl)
    ax.set_zlim(0, 1)

# bottom‐row aligned trajectories (columns 0..3 in the bottom row)
for idx, (Yc, lbl, cc) in enumerate(zip(aligned_cca, labels, correlation_coefficients)):
    ax = axes[1, idx]
    # plot original A’s canonical coords in green
    ax.plot(X_c[:, 1], X_c[:, 0], X_c[:, 2], color='tab:green', label='A')
    # plot aligned Yc in black
    ax.plot(Yc[:, 1], Yc[:, 0], Yc[:, 2], color='black', linestyle='-', label=f"{lbl}→A", alpha=0.75)
    ax.set_title(f"μ CC: {cc:.2f}")
    ax.set_zlim(0, 1)
    ax.legend(loc='upper right', fontsize=6)

axes[1, 4].axis('off')

fig.suptitle("Characteristics of CCA with Geometric Data", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # make room for the suptitle
plt.show()


# fig = plt.figure(figsize = (13,6))

# ax = fig.add_subplot(2,5,1, projection = '3d')
# ax.plot(A.T[1], A.T[0], A.T[2], label = "A")
# ax.legend(loc = 'upper right')

# for i, (raw, lbl) in enumerate(zip(datasets, labels)):
#     ax = fig.add_subplot(2, 5, i+2, projection = '3d')
#     ax.plot(raw.T[1], raw.T[0], raw.T[2], label = lbl)
#     ax.set_zlim(0,1)
#     ax.legend(loc = 'upper right')

# for i, (Yc, lbl, cc) in enumerate(zip(aligned_cca, labels, correlation_coefficients)):
#     ax = fig.add_subplot(2, 5, i+7, projection = '3d')
#     ax.plot(X_c.T[1], X_c.T[0], X_c.T[2], label = 'A')
#     ax.plot(Yc.T[1], Yc.T[0], Yc.T[2], label = f"{lbl} to A", alpha = 0.75)
#     ax.set_title(f"μ CC: {cc:.2f}")
#     ax.legend(loc = 'upper right')

# fig.suptitle("Characteristics of CCA with Geometric Data", fontsize=16)

# plt.show()


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