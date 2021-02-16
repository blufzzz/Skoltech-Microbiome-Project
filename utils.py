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
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.model_selection import KFold
from copy import copy

def get_neigh_perc(data, perc=95):
    perc_list = []
    for n_neighbors in np.arange(3, 30, 1):
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(data)
        neigborhood_X_dist, neigborhood_X_ind = nn.kneighbors(data, n_neighbors=n_neighbors)
        mean_neigh_distances = neigborhood_X_dist.mean(1)
        perc_list.append(np.percentile(mean_neigh_distances, perc))
    return perc_list

def create_clustering_pivot_table(results_list, methods_names, datasets_names, manifold='', noise_threshold=0.3, independent=True,
                                  not_indep_db_thresh=.8, not_indep_silh_thresh=0.6):
    all_results = dict(zip(methods_names, results_list))
    X_dbind =np.zeros((len(methods_names), len(datasets_names)))
    X_silh =np.zeros((len(methods_names), len(datasets_names)))

    for i,(method_name) in tqdm_notebook(enumerate(methods_names)): # dbscan, kmeans
        for j,(dataset_name) in enumerate(datasets_names):
            
            method_results_dict = all_results[method_name] # e.g. dbscan results
            names = method_results_dict.keys() # datasets names
            label = dataset_name 
            if len(manifold) > 0:
                label += '_' + manifold
            results_dict = method_results_dict[label]
            if len(results_dict) == 0:
                # no estimation
                n_dbind = 1
                n_silh = 1
            else:
                # filtering noise samples
                results_dict = {k:v for k,v in results_dict.items() if v[-1] < noise_threshold}
                if len(results_dict) > 0: # n_c: [db, sil, noise]
                    if independent:
                        n_dbind = sorted(results_dict, key=lambda x: results_dict[x][0])[0] # minimized
                        n_silh = sorted(results_dict, key=lambda x: results_dict[x][1])[-1] # maximized
                    else:
                        n_dbind, n_silh = 1,1
                        for k,v in results_dict.items():
                            if v[0] < not_indep_db_thresh and v[1] > not_indep_silh_thresh:
                                n_dbind,n_silh = k,k
                else:
                    print(f'{method_name} couldnt estimate N clusters for {label} given threshold: {noise_threshold}')
                    # too much noize
                    n_dbind = np.nan
                    n_silh = np.nan
            
            X_dbind[i,j] = n_dbind
            X_silh[i,j] = n_silh
    X_dbind = pd.DataFrame(data = X_dbind, columns=datasets_names, index=methods_names)
    X_silh = pd.DataFrame(data = X_silh, columns=datasets_names, index=methods_names)
    return X_dbind, X_silh



def clustering(datasets_dict, method_class, param_range, dbscan=False, dbscan_params_dict=None):
    # performing clustering
    cluster_metrics = defaultdict(dict)
    cluster_results = defaultdict(dict)

    for label, dataset in tqdm_notebook(datasets_dict.items()):
        
        if dbscan:
            min_eps, max_eps = dbscan_params_dict[label]
            param_range = np.linspace(min_eps*0.5, max_eps*1.5, len(param_range))
        
        for p in param_range:
            method = method_class(p)
            pred = method.fit_predict(dataset)
            if max(pred) > 0: # at least 2 clusters: [0,1]
                centers = cl_centers(dataset, pred)
                ind = davies_bouldin_score(dataset, pred)
                silh = silhouette_score(dataset, pred)
                n = len(np.unique(pred[pred != -1]))
                noise_ratio = sum(pred == -1)/len(pred)
                cluster_metrics[label][n] = [ind, silh, noise_ratio]
                cluster_results[label][n] = pred
            else:
                pass
#                 print(f'Only one cluster was found for {label}, method: {method.__class__.__name__} param: {p}')
    return cluster_metrics, cluster_results


def plot_proj_clustering(clustering_results, method='', suptitle=None):
    results = copy(clustering_results)
    L = len(clustering_results)
    if L == 3:
        # MERGED
        MERGED = True
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
        axes = [axes]
        datasets_iterator = enumerate([None])
    elif L == 12:
        # NOT MERGED
        MERGED = False
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20,15))
        datasets_iterator = enumerate(['ptb', 'AGP', 't2d', 'ibd'])
    else:
        raise RuntimeError()
    for i,dataset_name in datasets_iterator:
        for j,tax_name in enumerate(['o', 'f', 'g']):
            label = f'proj_{tax_name}' if MERGED else f'{dataset_name}_proj_{tax_name}'
            
            if len(method) > 0:
                label += '_' + method
            
            names = results.keys() # clustering_results = {}
            
            if label in names:
                data = results[label]
                if len(data) > 0:
                    df = pd.DataFrame(data=data).T
                    df.columns = ['Davies-Bouldin index', 'silhouette_score', 'noise_ratio']
                    df.sort_index(ascending=False, inplace=True)
                    if df['noise_ratio'].sum() == 0.:
                        df.drop('noise_ratio', axis=1, inplace=True)
                    ax = axes[i,j]
                    ax.set_xlabel('# estimated clusters')
                    ax.set_title(label)
                    df.plot.bar(ax=ax)
    
    plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16, color='blue')
    plt.show()

# sample the data via binomial mask
def sample_data(data, fraction):
    mask = np.random.binomial(1, fraction, data.shape[0]).astype(bool)
    return data[mask]

# Clusters centers
def cl_centers(X, pred, n_cl=None):
    if n_cl is None:
        n_cl = len(set(pred))
        if -1 in pred:
            n_cl -= 1
    centers = np.zeros((n_cl, X.shape[1]))
    for i in range(n_cl):
        centers[i] = X[pred == i].mean(0)
    return centers

def cross_val_score_custom(est, X_mf, X_pca, X_orig, scoring, cv):
    scoring_list = []
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X_mf):
        X_mf_train = X_mf[train_index]
        X_pca_train = X_pca[train_index]
        
        X_mf_test = X_mf[test_index]
        X_orig_test = X_orig[test_index]
        
        est.fit(X_mf_train, X_pca_train)
        X_orig_test_pred = est.predict(X_mf_test)
        
        scoring_list.append(scoring(X_orig_test, X_orig_test_pred))
    return np.array(scoring_list)


class MF2PCA2ORIG(BaseEstimator):
    def __init__(self, pca_module, mo_regressor, use_softmax=False):
        super(MF2PCA2ORIG, self).__init__()
        self.pca = pca_module
        self.mo_regressor = mo_regressor
        self.use_softmax = use_softmax
        
    def fit(self, X,y):
        self.mo_regressor.fit(X,y)
        return self
        
    def predict(self, X):
        y_pred = self.mo_regressor.predict(X)
        y_pred_orig = self.pca.inverse_transform(y_pred)
        if self.use_softmax:
            y_pred_orig = np.exp(y_pred_orig) / np.exp(y_pred_orig).sum(1)[:,None]
        return y_pred_orig


# Davies Bouldin Index
def DB_index(X, clusters_centers, labels):
    if -1 in labels:
        X = X[labels != -1]
        labels = labels[labels != -1]
    n_clusters = len(clusters_centers)
    d = np.array([distance.euclidean(X[i], clusters_centers[labels[i]]) for i in range(len(X))])
    mean_dist = np.zeros(n_clusters)
    for i in range(n_clusters):
        mean_dist[i] = d[labels == i].mean()
    return sum([max([(mean_dist[i] + mean_dist[j]) / distance.euclidean(clusters_centers[i], clusters_centers[j]) 
         for i in range(n_clusters) if i != j]) for j in range(n_clusters)]) / n_clusters

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


def NPR(X, Z, k=21):
    _, neigborhood_X = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    _, neigborhood_Z = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
    n = X.shape[0]
    npr = 0
    for i in range(n):
        npr += np.intersect1d(neigborhood_X[i], neigborhood_Z[i]).shape[0]
    npr_normed = npr / (k * n)
    return npr_normed