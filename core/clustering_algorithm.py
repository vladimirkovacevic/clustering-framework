import scanpy as sc
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, v_measure_score, mutual_info_score
from abc import ABC, abstractmethod


class ClusteringAlgorithm(ABC):
    def __init__(self, adata, **params):
        self.adata = adata
        self.adata.uns['algo_params'] = params
        self.adata.uns['sample_name'] = os.path.join(self.adata.uns['algo_params']['out_path'], os.path.basename(self.adata.uns['algo_params']['file'].rsplit(".", 1)[0]))
        for key, value in params.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save_results(self):
        pass

    def preprocess(self):
        self.adata.var_names_make_unique()
        self.adata.raw = self.adata
        sc.pp.filter_genes(self.adata, min_cells=10)
        sc.pp.normalize_total(self.adata, inplace=True)
        if "log1p" not in self.adata.uns_keys():
            sc.pp.log1p(self.adata)
        logging.info(f'Finished preprocessing')

    def plot_clustering_against_ground_truth(
        self,
        labels_true='annotation',
        labels_pred='clusters'
        ):
        # Remap obtained clustering to ground truth
        labels_pred = self.cluster_key
        labels_true = list(set(['celltype_pred','annotation']).intersection(set(self.adata.obs_keys())))[0]
        cm = contingency_matrix(labels_true=self.adata.obs[labels_true], labels_pred=self.adata.obs[labels_pred])
        cont_df = pd.DataFrame(cm, index=sorted(set(self.adata.obs[labels_true])), columns=sorted(set(self.adata.obs[labels_pred])))
        remap_dict = {col:cont_df.index[np.argmax(cont_df[col])] for col in cont_df.columns}
        self.adata.obs.loc[:, labels_pred + '_remap'] = self.adata.obs[labels_pred].map(remap_dict)

        remaped_classes = set(self.adata.obs[labels_pred + '_remap'])
        mask_col = [1 if x in remaped_classes else 0 for x in cont_df.index]
        colors = list(self.adata.uns[labels_true + '_colors'])
        remaped_colors = []
        for m, c in zip(mask_col, colors):
            if m:
                remaped_colors.append(c)

        # Plot ground truth
        self.plot_clustering(color=[labels_true], sample_name=f'{self.filename}_ground_truth.png', palette=colors)

        # Plot remaped clusters
        self.plot_clustering(color=[labels_pred + '_remap'], sample_name=f'{self.filename}.png', palette=remaped_colors)

    def plot_clustering(
        self,
        sample_name="unknown",
        color=['clusters'],
        palette=None
        ):
        color = [self.cluster_key]
        sc.pl.spatial(self.adata, color=color, palette=palette, spot_size=self.spot_size)
        plt.savefig(sample_name, dpi=200, bbox_inches='tight')
        plt.close()

    def calculate_clustering_metrics(self):
        res_dict = {}
        labels_true = list(set(['celltype_pred','annotation']).intersection(set(self.adata.obs_keys())))[0]
        for scorer, label in zip([adjusted_rand_score, v_measure_score, mutual_info_score], ['ARS', 'V-Score','Mut-info']):
            res = scorer(labels_true=self.adata.obs[labels_true], labels_pred=self.adata.obs[self.cluster_key])
            res_dict[label] = np.round(res, 3)
        logging.info(f'Calculated clustering metrics: {res_dict}')
        self.adata.uns['metrics'] = res_dict

