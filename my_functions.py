import numpy as np
from sklearn.decomposition import PCA # PCA reduces high dimensions
# from sklearn.cross_decomposition import CCA #aligning

# def align_cca(X, Y, n_components = None):
#     # computes CCA allignment between X and Y
#     if n_components is None:
#         n_components = min(X.shape[1], Y.shape[1])
#     cca = CCA(n_components = n_components)
#     X_c, Y_c = cca.fit_transform(X, Y)
#     return X_c, Y_c, cca

def CCA(A, B, align=None):
    """
    Canonical Correlation Analysis

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
    B : array-like of shape (n_samples, n_features)
    align: None, 'A2B', 'B2A'
        Align B to A or A to B
        if None, return None in aligned

    Returns
    -------
    cc: canonical correlations (n_features,)
    aligned: either B aligned to A or A aligned to B (n_samples, n_features)
        if align==None, then returns None
    """
    assert align in [None, 'A2B', 'B2A'], f'align should be one of None A2B, None], received {align}'

    Q_A, R_A = np.linalg.qr(A)
    Q_B, R_B = np.linalg.qr(B)
    U, S, V_T = np.linalg.svd(Q_A.T @ Q_B)

    if align == 'B2A':
        aligned = Q_B@V_T.T@U.T@R_A  # B aligned to A
    elif align == 'A2B':
        aligned = Q_A@U@V_T@R_B  # A aligned to B

    else:
        aligned = None
    return S, aligned