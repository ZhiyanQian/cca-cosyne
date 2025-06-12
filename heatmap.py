from my_functions import CCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import seaborn as sns

# HEATMAP 

trial_lengths = [100, 200, 300, 400, 500, 600, 700]
neurons = [20, 120, 220, 320, 420]
mean_cc = np.zeros((len(trial_lengths), len(neurons)))
firing_rate = 10 #average neurons fire at 10hz for moderate activity
dt = 0.001 # divide time (seconds) into bins

pspike = firing_rate * dt # probability that there is a spike in the time bin

for i, t_len in enumerate(trial_lengths):
    for j, neu in enumerate(neurons):
        reps = []
        for rep in range(10): # repeat 10 times to average
            #stimulate data
            X = (np.random.randn(t_len, neu) < pspike).astype(float)
            Y = (np.random.randn(t_len, neu) < pspike).astype(float)
            #gaussian filter to smooth
            Xsm = gaussian_filter1d(X, sigma = 10, axis = 0)
            Ysm = gaussian_filter1d(Y, sigma = 10, axis = 0)
            #pca
            Xp = PCA(n_components=3).fit_transform(Xsm)
            Yp = PCA(n_components=3).fit_transform(Ysm)
            #align Xp and Yp -> CCA
            S, aligne = CCA(Xp, Yp, align='B2A')
            #corelation coefficients
            reps.append(np.mean(S[:3]))
        mean_cc[i, j] = np.mean(reps) #avg of the cc after 10 reps

#using seaborn
plt.figure(figsize = (6, 6))
ax = sns.heatmap(mean_cc, annot = True, cmap = 'viridis', xticklabels=list(neurons), yticklabels=list(trial_lengths), cbar_kws={"label": "Mean CC"})
ax.set_title('Aligning Two Random Spikes')
ax.set_xlabel('# Neurons')
ax.set_ylabel('Trial Length (ms)')
plt.show()

#graph is weird. mean CC way too low -> prob because fake data. Graph is shorta inverted -> maybe because I messed up extent and origin?


# GRAPH

pcs = [5, 10, 15, 20, 25]
sigmas = [5, 20, 35]
threshold = 0.9995 # can I fix it so that mean CC isn't ungodly high

coordinate = {} #{sigma: {"pcs": [], "trial lengths": []}}
for sigma in sigmas:
    coordinate[sigma] = {"pcs": [], "trial lengths": []}

for sigma in sigmas:
    for pc in pcs:
        for t_len in trial_lengths:
            # generate X and Y dataset
            x_raw = np.random.randn(t_len, 100)
            y_raw = np.random.randn(t_len, 100)

            #gaussian filter smooth
            x_sm = gaussian_filter1d(x_raw, sigma = sigma, axis = 0)
            y_sm = gaussian_filter1d(y_raw, sigma = sigma, axis = 0)

            #pca
            x_p = PCA(n_components=pc).fit_transform(x_sm)
            y_p = PCA(n_components=pc).fit_transform(y_sm)

            #CCA
            S, aligned = CCA(x_p, y_p, align='B2A')

            #find correlation coefficients
            cc = S[0]

            if cc > threshold:
                coordinate[sigma]["pcs"].append(pc)
                coordinate[sigma]["trial lengths"].append(t_len)

#make sure each sigma has cc > 0.threshold
for sigma in sigmas:
    xs = coordinate[sigma]['pcs']
    ys = coordinate[sigma]['trial lengths']
    print(f"σ={sigma} → {len(xs)} points; unique PCs = {sorted(set(xs))}")

print(coordinate)

#graphing
plt.figure(figsize=(6,5))
for sigma in sigmas:
    x = np.array(coordinate[sigma]["pcs"])
    y = np.array(coordinate[sigma]["trial lengths"])

    if len(x) == 0: #not enough for line of best fit
        continue
    plt.scatter(x, y)

    #line of best fit
    coe = np.polyfit(x, y, 1)
    slope = coe[0]
    y_int = coe[1]

    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + y_int

    plt.plot(x_line, y_line, label = f"σ={sigma}")

plt.title("Experiments with mean CC > 0.9995")
plt.xlabel("# PCs")
plt.ylabel("Trial Lengths (ms)")
plt.legend()
plt.tight_layout()
plt.show()

# still really weird results with the graph