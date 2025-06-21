# when longleaf data recieved, repopulate
# replace simulate_spikes with real lab data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from itertools import product
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from my_functions import CCA

def simulate_spikes(patterns, t_len, n_neu, rate = 1000, dt = 0.001):
    """
    Creates fake simulated spikes with patterns:
    'poisson'
    'uniform'
    'burst': poission + periods of high burst
    """
    if patterns == 'poisson':
        p = rate * dt #probability
        return (np.random.randn(t_len, n_neu) < p).astype(bool)
    elif patterns == 'uniform':
        return np.random.randn(t_len, n_neu)
    elif patterns == 'burst':
        p = rate * dt
        X = (np.random.rand(t_len, n_neu) < p).astype(bool)
        # add bursts every 50 steps lasting 10 steps at p=0.5
        for start in range(0, t_len, 50):
            end = min(t_len, start + 10)
            X[start:end] += (np.random.rand(end-start, n_neu) < 0.5).astype(bool)
        return np.clip(X, 0, 1) # make sure between binary
    else:
        raise ValueError("Unknown pattern")
    
patterns = ['poisson', 'uniform', 'burst']
trial_length = 500
n_neurons = 200
pca_dims = [5, 10, 15] 
sigmas = [5, 20, 25]
reps = 20

results = []

for pattern, sigma, dim in product(patterns, sigmas, pca_dims):
    cca_vals = []
    for r in range(reps): # repeat reps time for more accurate data
        X_raw = simulate_spikes(pattern, trial_length, n_neurons)
        Y_raw = simulate_spikes(pattern, trial_length, n_neurons)

        # gaussian smoothing
        X_sm = gaussian_filter1d(X_raw, sigma = sigma, axis = 0)
        Y_sm = gaussian_filter1d(Y_raw, sigma = sigma, axis = 0)

        # pca reduction
        X_pca = PCA(n_components = dim).fit_transform(X_sm)
        Y_pca = PCA(n_components = dim).fit_transform(Y_sm)

        # cca aligning
        S, alignment  = CCA(X_pca, Y_pca, align = 'B2A')
        cca_vals.append(S[0]) # top correlation

    results.append({
        'pattern': pattern,
        'sigma': sigma,
        'pca_dim': dim,
        'cca_mean': np.mean(cca_vals),
        'cca_std:': np.std(cca_vals),
    })

df = pd.DataFrame(results)
print(df)

# line plot: CCA vs gaussian sigma
fig, ax = plt.subplots(figsize = (8, 6))

sns.lineplot(
    data = df,
    x = 'sigma', y = 'cca_mean',
    hue = 'pattern', style = 'pca_dim',
    markers=True, err_style='bars', ci='sd',
    ax = ax
)

ax.set_title('CCA vs Gaussian σ\nacross firing patterns & PCA dims')
ax.set_xlabel('Gaussian σ')
ax.set_ylabel('Mean top canonical correlation')

ax.legend(title='pattern / PCA dim', # moves legend outside of graph
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0)

fig.tight_layout()
plt.show()

# heatmap for each pattern

for pattern in patterns:
    sub = df[df['pattern'] == pattern].pivot(index='sigma', columns='pca_dim', values='cca_mean')
        
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        sub,
        annot=True, fmt=".2f",
        cmap="viridis",
        cbar_kws={'label': 'Mean CC'},
        ax=ax
    )
    ax.set_title(f"{pattern} pattern")
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Gaussian σ')
        
    fig.tight_layout()
    plt.show()