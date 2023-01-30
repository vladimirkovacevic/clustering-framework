from sklearn.metrics import adjusted_rand_score, v_measure_score, mutual_info_score
import scanpy as sc
import numpy as np


def calculate_clustering_metrics(adata:sc.AnnData, labels_true='annotation', labels_pred='clusters'):
    res_dict = {}
    for scorer, label in zip([adjusted_rand_score, v_measure_score, mutual_info_score], ['ARS', 'V-Score','Mut-info']):
        res = scorer(labels_true=adata.obs[labels_true], labels_pred=adata.obs[labels_pred])
        res_dict[label] = np.round(res, 3)
    return res_dict
