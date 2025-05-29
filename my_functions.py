import numpy as np
from sklearn.decomposition import PCA # PCA reduces high dimensions
from sklearn.cross_decomposition import CCA #aligning

def align_cca(X, Y, n_components = None):
    # computes CCA allignment between X and Y
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])
    cca = CCA(n_components = n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    return X_c, Y_c, cca