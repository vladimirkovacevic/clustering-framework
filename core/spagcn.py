import sys
import os,csv,re
import argparse
import time

import pandas as pd
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import SpaGCN as spg
from core import ClusteringAlgorithm


class SpagcnAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        # SVG identification with SpaGCN
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + '_spagcn'
        self.cluster_key = 'spagcn'

    def run(self):
        self.preprocess()
        spatial_key = 'spatial' if 'spatial' in self.adata.obsm_keys() else ['x', 'y'] if set(['x', 'y']) <= set(self.adata.obs_keys()) else None

        if not spatial_key:
            raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
        start = time.perf_counter()

        if not self.svg_only:
            #Set coordinates
            self.adata.obs["x_array"]=self.adata.obsm['spatial'][:, 0]
            self.adata.obs["y_array"]=self.adata.obsm['spatial'][:, 1]
            self.adata.obs["x_pixel"]=self.adata.obsm['spatial'][:, 0]
            self.adata.obs["y_pixel"]=self.adata.obsm['spatial'][:, 1]

            x_array=adata.obs["x_array"].tolist()
            y_array=adata.obs["y_array"].tolist()
            x_pixel=adata.obs["x_pixel"].tolist()
            y_pixel=adata.obs["y_pixel"].tolist()
            #Run SpaGCN
            adata.obs[self.cluster_key]= spg.detect_spatial_domains_ez_mode(adata, None, x_array, y_array, x_pixel, y_pixel, n_clusters=args.max_num_clusters, histology=False, s=1, b=49, p=0.5, r_seed=100, t_seed=100, n_seed=100)
            adata.obs[self.cluster_key]=adata.obs[self.cluster_key].astype('category')
            #Refine domains (optional)
            if self.spagcn___refine:
                self.adata.obs[self.cluster_key]=spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), x_array=x_array, y_array=y_array, shape="hexagon")
                self.adata.obs[self.cluster_key]=self.adata.obs["refined_pred"].astype('category')
            end_domaining = time.perf_counter()
            logging.info(f'SpaGCN domain calculation took: {end_domaining - start} sec.')

        domains = set(self.adata.obs[self.cluster_key].values)
        logging.info(f'Found domains: {domains}')
        # Find SVGs
        self.adata.X=(adata.X.A if issparse(adata.X) else adata.X)

        #Set filtering criterials
        min_in_group_fraction=0.8
        min_in_out_group_ratio=1
        min_fold_change=1.5
        all_filtered_info = pd.DataFrame()
        for target in domains:
            logging.info(f'Processing domain {target}...')
            filtered_info=spg.detect_SVGs_ez_mode(self.adata, target=target, x_name="x_array", y_name="y_array", domain_name="pred", min_in_group_fraction=min_in_group_fraction, min_in_out_group_ratio=min_in_out_group_ratio, min_fold_change=min_fold_change)
            if len(filtered_info) > 0:
                # If zero genes found for the domain
                filtered_info.loc[:, 'domain'] = int(target)
            all_filtered_info = pd.concat([all_filtered_info, filtered_info]) if len(all_filtered_info) > 0 else filtered_info
            logging.info(f'Found {len(filtered_info)} DEGs for domain {target}')

        self.adata.uns['svg_' + self.cluster_key] = all_filtered_info
        
        end = time.perf_counter()
        logging.info(f'SpaGCN clustering done. Added results to adata.obs[\'{self.cluster_key}\']')
        logging.info(f'Total execution time: {end - start} sec.')

    def save_results(self):
        if self.method == 'all':
            self.filename = self.adata.uns['sample_name'] + "_all"
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        self.adata.uns['svg_' + self.cluster_key].to_csv(self.filename + '_svg.csv', index=False)
        logging.info(f'Saved processing result {self.filename}.h5ad.')