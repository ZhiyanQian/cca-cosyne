from my_functions import align_cca
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

trial_lengths = [100, 200, 300, 400, 500, 600, 700]
neurons = [20, 120, 220, 320, 420]
mean_cc = np.zeros((len(trial_lengths), len(neurons)))

for i, t_len in enumerate(trial_lengths):
    for j, neu in enumerate(neurons):
        reps = []
        for rep in range(10): # repeat 100 times to average
            #stimulate data
            X = np.random.randn(t_len, neu)
            Y = np.random.randn(t_len, neu)
            #pca
            Xp = PCA(n_components=3).fit_transform(X)
            Yp = PCA(n_components=3).fit_transform(Y)
            #align Xp and Yp -> CCA
            X_c, Y_c, __ = align_cca(Xp, Yp)
            #corelation coefficients
            r_vals = [np.corrcoef((X_c[:, k], Y_c[:, k]))[0, 1] for k in range(3)]
            reps.append(np.mean(r_vals)) #appends 10 cc at end of loop
        mean_cc[i, j] = np.mean(reps) #avg of the cc after 10 reps

ax = plt.figure(figsize=(6,6))
im = ax.imshow(mean_cc, origin = 'upper', aspect = 'auto', extent=[min(neurons), max(neurons), max(trial_lengths), min(trial_lengths)])
plt.colorbar(label = "Mean CC [0, 1]")
for i in range(len(neurons)):
    for j in range(len(trial_lengths)):
        value = mean_cc[i, j]
        ax.text(j, i, f"{value:.2f}")
plt.title('Aligning Two Random Spikes')
plt.xlabel('# Neurons')
plt.ylabel('Trial Length (ms)')
plt.gca().set_aspect('equal') #how to make square???? and how to center tick marks???
plt.show()