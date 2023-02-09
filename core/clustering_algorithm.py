import logging
import os

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, v_measure_score, mutual_info_score
from abc import ABC, abstractmethod

NUM_MARKER_GENES = 15

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
        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=3)
        sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
        if "log1p" not in self.adata.uns_keys():
            sc.pp.log1p(self.adata)
        # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5) TODO discuss if needed
        self.adata.raw = self.adata
        # sc.pp.regress_out(adata, ['total_counts'])  TODO discuss if needed
        # sc.pp.scale(adata, max_value=10)  TODO discuss if needed
        logging.info(f'Finished preprocessing')

    def plot_clustering_against_ground_truth(
        self,
        labels_true='annotation',
        labels_pred='clusters'
        ):
        labels_pred = self.cluster_key
        labels_true = list(set(['celltype_pred','annotation']).intersection(set(self.adata.obs_keys())))[0]
        remapped_colors = self.remap_colors_to_ground_truth_by_labels(labels_true, labels_pred)
        # Plot ground truth
        self.plot_clustering(color=[labels_true], sample_name=f"{self.adata.uns['sample_name']}_ground_truth.png", palette=list(self.adata.uns[labels_true + '_colors']))
        # Plot remaped clusters
        self.plot_clustering(color=[labels_pred + '_remap'], sample_name=f'{self.filename}.png', palette=remapped_colors)

    def plot_tissue_modules_against_ground_truth(
        self,
        labels_true='annotation',
        labels_pred='clusters'
        ):
        labels_pred = self.cluster_key
        labels_true = list(set(['celltype_pred','annotation']).intersection(set(self.adata.obs_keys())))[0]
        marker_gene_key_true = self.identify_marker_genes(labels_true)
        marker_gene_key_pred = self.identify_marker_genes(labels_pred)
        self.decide_tm_via_marker_genes(marker_gene_key_true, marker_gene_key_pred)

        # Plot ground truth
        self.plot_clustering(color=[labels_true], sample_name=f"{self.adata.uns['sample_name']}_ground_truth.png", palette=list(self.adata.uns[labels_true + '_colors']))
        # Plot remaped clusters
        self.plot_clustering(color=[marker_gene_key_pred + '_tm_remap'], sample_name=f'{self.filename}_tm_.png', palette=list(self.adata.uns[labels_true + '_colors']))
    
    def plot_clustering(
        self,
        sample_name="unknown",
        color=['clusters'],
        palette=None
        ):
        figure, ax = plt.subplots(nrows=1, ncols=1)
        sc.pl.spatial(self.adata, color=color, palette=palette, spot_size=self.spot_size, ax=ax, show=False)
        figure.savefig(sample_name, dpi=200, bbox_inches='tight')
        plt.close()

    def calculate_clustering_metrics(self):
        res_dict = {}
        labels_true = list(set(['celltype_pred','annotation']).intersection(set(self.adata.obs_keys())))[0]
        for scorer, label in zip([adjusted_rand_score, v_measure_score, mutual_info_score], ['ARS', 'V-Score','Mut-info']):
            res = scorer(labels_true=self.adata.obs[labels_true], labels_pred=self.adata.obs[self.cluster_key])
            res_dict[label] = np.round(res, 3)
        logging.info(f'Calculated clustering metrics: {res_dict}')
        self.adata.uns['metrics'] = res_dict

    def remap_colors_to_ground_truth_by_labels(
        self,
        labels_true: str,
        labels_pred: str
        ):
        # Remap obtained clustering to ground truth colors
        cm = contingency_matrix(labels_true=self.adata.obs[labels_true], labels_pred=self.adata.obs[labels_pred])
        cont_df = pd.DataFrame(cm, index=sorted(set(self.adata.obs[labels_true])), columns=sorted(set(self.adata.obs[labels_pred])))
        remap_dict = {col:cont_df.index[np.argmax(cont_df[col])] for col in cont_df.columns}
        self.adata.obs.loc[:, labels_pred + '_remap'] = self.adata.obs[labels_pred].map(remap_dict)

        remapped_classes = set(self.adata.obs[labels_pred + '_remap'])
        mask_col = [1 if x in remapped_classes else 0 for x in cont_df.index]
        colors = list(self.adata.uns[labels_true + '_colors'])
        remapped_colors = []
        for m, c in zip(mask_col, colors):
            if m:
                remapped_colors.append(c)
        return remapped_colors
    
    def decide_tm_via_marker_genes(
        self,
        labels_true: str,
        labels_pred: str
        ):
        tm_to_mg_actual_dict = {name: set(self.adata.uns[labels_true]['names'][name][:NUM_MARKER_GENES]) for name in self.adata.uns[labels_true]['names'].dtype.names}
        tm_to_mg_predicted_dict = {name: set(self.adata.uns[labels_pred]['names'][name][:NUM_MARKER_GENES]) for name in self.adata.uns[labels_pred]['names'].dtype.names}

        def tissue_module_max_intersect(marker_genes_predicted):
            num_intersecting_genes = {ground_truth_module: len(list(set(marker_genes_predicted).intersection(marker_genes_actual))) \
                for ground_truth_module, marker_genes_actual in tm_to_mg_actual_dict.items()}
            max_intersecting_module = max(num_intersecting_genes.items(), key=lambda kv: kv[1])[0]
            return max_intersecting_module
        
        remap_predicted_dict = {k: tissue_module_max_intersect(v) for k, v in tm_to_mg_predicted_dict.items()}
        self.adata.obs.loc[:,labels_pred + '_tm_remap'] = self.adata.obs[self.cluster_key].map(remap_predicted_dict)

    def identify_marker_genes(
        self,
        groupby: str
        ):
        if groupby == 'leiden':
            logging.info("Leiden clustering will be performed in order to generate the key of the observations for finding the marker genes")
            sc.tl.pca(self.adata, svd_solver='arpack')
            sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=40)
            sc.tl.leiden(self.adata)
        key_added = f"{groupby}_marker_genes"
        sc.tl.rank_genes_groups(self.adata, groupby, method='t-test', key_added=key_added)
        return key_added
