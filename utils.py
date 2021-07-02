import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np 
from numpy import genfromtxt
from numpy import linalg as LA
import warnings
warnings.filterwarnings('ignore') 
from collections import defaultdict, Counter
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances, make_scorer
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from itertools import combinations
from copy import copy
from numba import cuda, jit, njit, prange, vectorize
from sklearn.model_selection import cross_val_score, cross_validate, ParameterGrid, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor, RegressorChain, RegressorMixin
from coranking import coranking_matrix
from coranking.metrics import continuity, LCMC, trustworthiness

RANDOM_SEED=42
FIGSIZE=(5,5)
DPI=150
FONTSIZE=12

SILHOETTE_THRESHOLD_DISTINCT = 0.7
DB_THRESHOLD_DISTINCT = 0.4

def transform(method, est, X, parameters, scorer):
    model_inst = method(n_components=2, **parameters, n_jobs=-1)
    X_trans = model_inst.fit_transform(X)
    if np.isnan(X_trans).any():
        return np.inf
    else:
        cv_score = -cross_val_score(est, X_trans, X, scoring=scorer, cv=3, n_jobs=-1).mean()
        return cv_score

    
def NN_ratio_score(X,y, k=5):
    X_pd = squareform(pdist(X))
    s = []
    s_min = []
    for y_i in np.unique(y):
        mask = y==y_i
        nearest_mates = np.sort(X_pd[mask,:][:,mask], axis=1)[:,1:k+1].mean(1)
        nearest_foes = np.sort(X_pd[mask,:][:,~mask], axis=1)[:,1:k+1].mean(1) + 1e-5

    #     y_i_diam = X_pd[mask,:][:,mask].min() + 1e-5
    #     y_ij_r = X_pd[mask,:][:,~mask].max() + 1e-5

    #     s_min.append(y_ij_r/y_i_diam)

        s.append((nearest_mates/nearest_foes).mean())
    return np.mean(s)# / max(s_min)
    
# def nearest_ratio_score(X,y, n_neighbors=5):
#     nn = NearestNeighbors(n_neighbors=n_neighbors)
#     nn.fit(X)
#     neigh_dist, neigh_ind = nn.kneighbors(X)
#     s = []
#     for y1,y2 in list(combinations(np.unique(y),2)):
#         # take two different clusters
#         dist_C0 = neigh_dist[:,1:][y==y1].mean(1) # ср расстояние до ближайшей из своего кластера
#         dist_C1 = neigh_dist[:,1:][y==y2].mean(1) # ср расстояние до ближайшей из своего кластера
    
#         cdist_C = cdist(X[y==y1],X[y==y2])
        
#         cdist_C.sort(axis=1)
#         cdist_C0 = cdist_C[:,:n_neighbors].mean(axis=1) # ср расстояние до ближайшей из чужого кластера
#         cdist_C.sort(axis=0)
#         cdist_C1 = cdist_C[:n_neighbors].mean(axis=0) # ср расстояние до ближайшей из чужого кластера
        
#         s0 = (dist_C0/cdist_C0).mean()
#         s1 = (dist_C1/cdist_C1).mean()
#         s.append((s0 + s1)/2)
#     return np.mean(s)

def norm_entropy(x):
    return -(x*np.log(x)).sum() / len(x)

def unpack_data(paths):
    datasets = {}
    for path in paths:
        label = path.split('/')[-1].split('.')[0].split('_')[:3]
        label = '_'.join(label)
        try:
            datasets[label] = np.genfromtxt(path, delimiter=';')
        except:
            datasets[label] = np.load(path, allow_pickle=True)
    return datasets

def mae_score(y_pred, y):
    return (np.linalg.norm(y_pred - y, axis=1, ord=1)/ (np.linalg.norm(y, axis=1, ord=1) + 1e-7)).mean()

def unpack_data(paths):
    datasets = {}
    for path in paths:
        label = path.split('/')[-1].split('.')[0].split('_')[:3]
        label = '_'.join(label)
        try:
            datasets[label] = np.genfromtxt(path, delimiter=';')
        except:
            datasets[label] = np.load(path, allow_pickle=True)
    return datasets

def plot_results(DISTINCT_CLUSTERS_RESULTS, data, ml_method_name):
    
    for label, preds_dict in DISTINCT_CLUSTERS_RESULTS.items():
        for method_name, (preds, n_clusters) in preds_dict.items():
            dataset = data[label]
            dataset_name = label.split('_')[0]
            tax_name = label.split('_')[-1].upper()
            # TSNE
            N = min(dataset.shape[1], 5)
#             fig, axes = plt.subplots(ncols=N+1, nrows=1, figsize=(5*(N+1), 5), dpi=280)
            
            tsne2 = TSNE(2, init='pca', verbose=0, angle=0.3, perplexity=20, n_iter=500, random_state=RANDOM_SEED)
            tsne2.fit(dataset)
            embedding = tsne2.embedding_

            colors = [cm.get_cmap('viridis')(it) for it in np.linspace(0,1,n_clusters)]
#             for c_number in range(n_clusters):
#                 mask = preds[n_clusters] == c_number
#                 perc = round(mask.sum()/len(mask),3)
#                 axes[0].scatter(embedding[:,0][mask], embedding[:,1][mask], alpha=0.3, c=colors[c_number], label=f'cluster {c_number+1}, {perc}% of data')
#             axes[0].set_title(f'Dataset: {dataset_name}, Tax: {tax_name}, Maifold Learning method: LLE \n t-SNE 2D visualization, {method_name} clustering to {n_clusters} clusters',fontsize=FONTSIZE)
#             axes[0].legend()
            # PROJECTIONS
            plt.figure(figsize=FIGSIZE, dpi=DPI)
            for c_number in range(n_clusters):
                mask = preds[n_clusters] == c_number
                perc = round(mask.sum()/len(mask),3)
                plt.scatter(embedding[:,0][mask], embedding[:,1][mask], alpha=0.3, c=colors[c_number], label=f'cluster {c_number+1}, {perc}% of data')
            plt.title(f'Dataset: {dataset_name}, Tax: {tax_name}, Maifold Learning method: {ml_method_name} \n t-SNE 2D visualization, {method_name} clustering to {n_clusters} clusters',fontsize=FONTSIZE)
            plt.legend()
            plt.show()
#             for i,(d1,d2) in enumerate(list(combinations(np.arange(N),2))[:N]):
#                 axes[i+1].scatter(dataset[:,d1], dataset[:,d2], alpha=0.1, c=preds[n_clusters])
#                 axes[i+1].set_title(f'{dataset_name}, DIMS:{d1,d2}, {method_name} clustering to n_clust={n_clusters}')
#             clear_output()
#             plt.tight_layout()
#             plt.show()



def get_report(methods_dict, 
               cluster_results_list, 
               cluster_preds_list,
               silh_thresh=0.8, 
               dbind_thresh=1.5, 
               noise_thresh=0.3,
               H_thresh=0.3,
               n_clusters_max=5,
               data_default_metrics_max_silhoette=None):
    
    report_results = defaultdict(dict)
    for index, method_name in enumerate(methods_dict.keys()):
        print('------------------------------')
        print(method_name)
        for dataset_label, dataset_results in cluster_results_list[index].items():
            for n_clusters, metrics in dataset_results.items():
                ind, silh, noise_ratio = metrics
                preds = cluster_preds_list[index][dataset_label][n_clusters]
                if data_default_metrics_max_silhoette is not None:
                    silh_thresh = data_default_metrics_max_silhoette[dataset_label][1]
                if silh >= silh_thresh and ind <= dbind_thresh and noise_ratio <= noise_thresh:
                    ratios = np.array(list(Counter(preds).values())) / len(preds)
                    H = -(ratios*np.log(ratios)).sum()
                    ratios = ratios.round(3)
                    if len(ratios) < n_clusters_max and H > H_thresh:
                        print(f'Ratio for {dataset_label}, for n_clusters={n_clusters}, H={H}, ratios={ratios}, DBind={ind}, Silh={silh}')
                        
        print('------------------------------')

def filter_paths(paths, keywords=[]):
    paths_filtered = []
    for word in keywords:
        paths_filtered += list(filter(lambda x: word in x.split("/")[-1].split(".")[0], paths))
    return paths_filtered

@njit
def js(p,q):
    EPS = 1e-10
    dkl_pq = np.sum(p * np.log((p+EPS)/(q + EPS)))
    dkl_qp = np.sum(q * np.log((q+EPS)/(p + EPS)))
    J = (dkl_pq + dkl_qp)/2
    return J

def get_neigh_perc(data, perc=95, metric='minkowski'):
    perc_list = []
    for n_neighbors in np.arange(3, 30, 1):
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(data)
        neigborhood_X_dist, neigborhood_X_ind = nn.kneighbors(data, n_neighbors=n_neighbors)
        mean_neigh_distances = neigborhood_X_dist[:,1:].mean(1)
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


def clustering(datasets_dict, method_class, param_range, precomputed=False, dbscan=False, dbscan_params_dict=None):
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
                
                ind = davies_bouldin_score(dataset, pred)
                nnrs = NN_ratio_score(dataset,pred)
                silh = silhouette_score(dataset, pred)
                
                unique_clusters = np.unique(pred[pred != -1])
                n = len(unique_clusters)
                cl_dist = np.ones(n)
                for i,cl_number in enumerate(unique_clusters):
                    cl_dist[i] = sum(pred == cl_number)/len(pred)
                
                noise_ratio = sum(pred == -1)/len(pred)
                norm_entropy_ = norm_entropy(cl_dist)
                
                if n in cluster_metrics[label]:
                    # same clustering with lower silhoette score
                    if silh < cluster_metrics[label][n][1]:
                        continue
                        
                cluster_metrics[label][n] = [ind, silh, noise_ratio, norm_entropy_, nnrs] # ,nrs
                cluster_results[label][n] = pred
            else:
                pass
#                 print(f'Only one cluster was found for {label}, method: {method.__class__.__name__} param: {p}')
    return cluster_metrics, cluster_results


def plot_clustering(clustering_results, method='', suptitle=None, data_default_metrics=None):
    '''
    clustering_results - dict
    '''
    results = copy(clustering_results)
    L = len(clustering_results)
    
    for label,data in results.items():
        if len(data) > 0:
            plt.figure(figsize=FIGSIZE, dpi=DPI)
            df = pd.DataFrame(data=data).T
            df.columns = ['Davies-Bouldin index', 'Silhouette score', 'Noise Ratio', 'Normalized Entropy', 'NN Ratio Score'] # , 'NN Ratio Score'
#             df.drop('NN Ratio Score', axis=1, inplace=True)
            df.sort_index(ascending=False, inplace=True)
#             colors = ['blue', 'orange', 'green', 'red', 'purple']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'] # 

            if df['Noise Ratio'].sum() == 0.:
                df.drop('Noise Ratio', axis=1, inplace=True)
                colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple'] # 
            plt.xlabel('# estimated clusters', fontsize=FONTSIZE)
            dataset = label.split('_')[0]
            tax = label.split('_')[-1]
            tax = tax.capitalize()
            if dataset == 'ptb':
                dataset = 'HMP'
            label = f'Dataset: {dataset}, Tax: {tax}'
            label = label + '\n' + method if len(method) > 0 else label
            plt.title(suptitle + ', ' + label if suptitle is not None else label, fontsize=FONTSIZE)
#             plt.plot(df.index.to_list(), [DB_THRESHOLD_DISTINCT]*df.shape[0], linestyle='--', color='blue')
            plt.hlines(DB_THRESHOLD_DISTINCT, -1, max(df.index), linestyle='--', color='blue', alpha=0.5, label='DB index threshold')
            plt.hlines(SILHOETTE_THRESHOLD_DISTINCT, -1, max(df.index), linestyle='--', color='orange', alpha=0.5, label='Silhoette score threshold')
            plt.legend(fontsize=6)
            df.plot.bar(ax=plt.gca(), color=colors)

            if data_default_metrics is not None:
                def_clust_type, Silhoette_default = data_default_metrics[label]
    #                         if not DB_default > 1.5*max(df['Davies-Bouldin index']):
    #                             ax.hlines(DB_default, 0, len(df.index), linestyles='dotted', colors='blue')
                ax.hlines(Silhoette_default, 0, len(df.index), linestyles='dotted', colors='orange', label=def_clust_type)
                ax.legend(fontsize=6)
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
        fig = plt.figure(figsize=(10,5), dpi=DPI)

#         plt.subplot(121)
#         plt.title("Explained variance", fontsize=FONTSIZE)
#         plt.xlabel("Number of PCs", fontsize=FONTSIZE)
#         plt.grid(linestyle="dotted")
#         plt.plot(EV, "o-")

# #         plt.subplot(122)
        plt.title("Cumulative explained variance", fontsize=FONTSIZE)
#         plt.axhline(linewidth=1, y=0.95, color='r')
#         plt.axhline(linewidth=1, y=0.9, color='r')
#         plt.axhline(linewidth=1, y=0.8, color='r')
        plt.xlabel("Number of PCs", fontsize=FONTSIZE)
        plt.grid(linestyle="dotted")
        plt.plot(CEV, "o-")        
        plt.axhline(linewidth=1, y=0.99, color='r', label='99% of explained variance')
        plt.legend(fontsize=FONTSIZE)
            
    return ev_num
    
def project(data, plot=False):
    pca = PCA(svd_solver='full',random_state=42)
    pca.fit(data)
    ev_num = get_threshold_eigennumbers(pca, data, plot=plot)
    data_centered = data - data.mean(0)[None,...]
    pca_proj = PCA(n_components=ev_num,svd_solver='full',random_state=42)
    data_projected = pca_proj.fit_transform(data)
    if plot:
        print('EV_NUM',ev_num,"REC_ERROR:", mae_score(pca_proj.inverse_transform(data_projected), data_centered))
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



def transform(method, X, dim, parameters, scorer):
    try:
        model_inst = method(n_components=dim, **parameters, n_jobs=-1)
        Z = model_inst.fit_transform(X)
    except:
        print(f'ERROR for {parameters}')
    if np.isnan(Z).any():
        print(f'NAN for {parameters}')
        return None
    else:
        return scorer(X,Z)


def cross_validate(X,Z):
    knn = KNeighborsRegressor()
    mo = MultiOutputRegressor(knn)
    mae_scorer = make_scorer(mae_score, greater_is_better=False)
    return -cross_val_score(mo, Z, X, scoring=mae_scorer, cv=3, n_jobs=-1).mean()



def calculate_Q_mae(X, Z):
    
    Q = coranking_matrix(X, Z)
    
    m = X.shape[0]
    UL_cumulative = 0 
    Q_k = []
    LCMC_k = []
    for k in range(0, Q.shape[0]):
        r = Q[k:k+1,:k+1].sum()
        c = Q[:k,k:k+1].sum()
        UL_cumulative += (r+c)
        Qnk = UL_cumulative/((k+1)*m) # (k+1)
        Q_k.append(Qnk)
        LCMC_k.append(Qnk - ((k+1)/(m-1)))
    
    argmax_k = np.argmax(LCMC_k)
    k_max = np.arange(1.,m)[argmax_k]
    Q_loc = (1./k_max)*np.sum(Q_k[:argmax_k+1])
    Q_glob = (1./(m-k_max))*np.sum(Q_k[argmax_k+1:])
    
    mae = cross_validate(X, Z)
    
    return Q_loc, Q_glob, mae