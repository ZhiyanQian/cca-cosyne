from my_functions import align_cca
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import seaborn as sns

# HEATMAP 

trial_lengths = [100, 200, 300, 400, 500, 600, 700]
neurons = [20, 120, 220, 320, 420]
mean_cc = np.zeros((len(trial_lengths), len(neurons)))

for i, t_len in enumerate(trial_lengths):
    for j, neu in enumerate(neurons):
        reps = []
        for rep in range(10): # repeat 10 times to average
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

# plt.imshow(mean_cc, origin = 'upper', aspect = 'equal', extent=[min(neurons), max(neurons), max(trial_lengths), min(trial_lengths)])
# plt.colorbar(label = "Mean CC [0, 1]")
# # for i in range(len(neurons)): # why no display CC within box
# #     for j in range(len(trial_lengths)):
# #         value = mean_cc[i, j]
# #         plt.text(j, i, f"{value:.2f}")
# plt.title('Aligning Two Random Spikes')
# plt.xlabel('# Neurons')
# plt.ylabel('Trial Length (ms)')
# plt.gca().set_aspect('equal') #how to make square???? and how to center tick marks???
# plt.show()

#using seaborn
sns.heatmap(mean_cc, annot = True, cmap = 'viridis', xticklabels=list(neurons), yticklabels=list(trial_lengths))
plt.title('Aligning Two Random Spikes')
plt.xlabel('# Neurons')
plt.ylabel('Trial Length (ms)')
plt.show()

#graph is weird. mean CC way too low -> prob because fake data. Graph is shorta inverted -> maybe because I messed up extent and origin?


# GRAPH

pcs = [5, 10, 15, 20, 25]
sigmas = [5, 20, 35]

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
            X_c, Y_c, __ = align_cca(x_p, y_p, n_components=pc)

            #find correlation coefficients
            cc = np.corrcoef(X_c[:,0], Y_c[:,0])[0, 1]

            if cc > 0.12: #using 0.1 for now
                coordinate[sigma]["pcs"].append(pc)
                coordinate[sigma]["trial lengths"].append(t_len)

#make sure each sigma has cc > 0.12
for sigma in sigmas:
    count = len(coordinate[sigma]["pcs"])
    print(f"σ = {sigma} → collected {count} points that pass CC > {0.1}") 

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

plt.title("Experiments with mean CC > 0.1")
plt.xlabel("# PCs")
plt.ylabel("Trial Lengths (ms)")
plt.legend()
plt.tight_layout()
plt.show()

#why weird graph? why only sigma 35 showing -> is everything overlapping?