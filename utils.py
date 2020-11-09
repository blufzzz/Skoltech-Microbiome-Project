import pandas as pd
from sklearn.decomposition import PCA
import numpy as np 
from numpy import genfromtxt
from numpy import linalg as LA
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt


def EV_i(i, eigenvalues):
    return eigenvalues[i] / np.sum(eigenvalues)

def CEV_d(d, eigenvalues):
    eigenvalues_d_sum = 0
    for i in range(d):
        eigenvalues_d_sum += eigenvalues[i]
    return eigenvalues_d_sum / np.sum(eigenvalues)

def get_threshold_eigennumbers(pca, data, threshold=0.99, plot=False):
    # define the i-th normalized eigenvalue - expected variance
    # calculating the list of the EVs and CEVs
    # evnum is the first eigenvalue having CEV more or equal to 0.99
    EV = []
    CEV = []
    ev_num = -1
    
    explained_variance = pca.explained_variance_
    for i in range(explained_variance.shape[0]):
        EV.append(EV_i(i, explained_variance))
        CEV.append(CEV_d(i, explained_variance))
        if (ev_num == -1 and CEV[i] >= threshold):
            ev_num = i
    #plot EV/CEVs
    if plot:
        fig = plt.figure(figsize=(12,5.25))

        plt.subplot(121)
        plt.title("Explained variance")
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.plot(EV, "o-")

        plt.subplot(122)
        plt.title("Cumulative explained variance")
        plt.axhline(linewidth=1, y=0.99, color='r')
        plt.axhline(linewidth=1, y=0.95, color='r')
        plt.axhline(linewidth=1, y=0.9, color='r')
        plt.axhline(linewidth=1, y=0.8, color='r')
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.plot(CEV, "o-")        
            
    return ev_num
    
def project(data, plot=False):
    pca = PCA(svd_solver='full',random_state=42)
    pca.fit(data)
    ev_num = get_threshold_eigennumbers(pca, data, plot=plot)
    data_centered = data - data.mean(0)[None,...]
    pca_proj = PCA(n_components=ev_num,svd_solver='full',random_state=42)
    data_projected = pca_proj.fit_transform(data)
    if plot:
        print('EV_NUM',ev_num,"REC_ERROR:",np.abs(pca_proj.inverse_transform(data_projected) - data_centered).mean())
    return data_projected, pca_proj