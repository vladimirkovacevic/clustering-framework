import argparse as ap
import logging
import os
import sys

import SpaGFT as spg
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scipy

from sklearn.cluster import spectral_clustering
from core import ClusteringAlgorithm
from .utils import timeit

class SpagftAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_spagft_r{self.resolution}_rl{self.spagft__ratio_low_freq}_rh{self.spagft__ratio_high_freq}_rn{self.spagft__ratio_neighbors}__mg{self.n_marker_genes}"
        self.cluster_key = 'spagft'

    @timeit
    def run(self):
        if 'tm_pseudo_expression' not in self.adata.obsm_keys():
            self.preprocess()
            spatial_key = 'spatial' if 'spatial' in self.adata.obsm_keys() else ['x', 'y'] if set(['x', 'y']) <= set(self.adata.obs_keys()) else None

            if not spatial_key:
                raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
            # find SVGs
            spg.rank_gene_smooth(self.adata,
                                ratio_low_freq=self.spagft__ratio_low_freq,
                                ratio_high_freq=self.spagft__ratio_high_freq,
                                ratio_neighbors=self.spagft__ratio_neighbors,
                                spatial_info=spatial_key)
            logging.info(f'Identified spatially variable genes')
            if self.svg_only:
                return
            # identify tissue modules
            spg.gft.find_tissue_module(self.adata, 
                                        ratio_fms=self.spagft__ratio_fms,
                                        ratio_neighbors=self.spagft__ratio_neighbors,
                                        spatial_info=spatial_key,
                                        quantile=self.spagft__quantile,
                                        resolution=self.resolution
                                        )
            logging.info(f'Identified tissue modules')
            self.adata.obsm['tm_pseudo_expression_val'] = self.adata.obsm['tm_pseudo_expression'].values
            sc.pp.neighbors(self.adata, 
                            n_neighbors=self.spagft__n_neighbors, 
                            n_pcs=len(self.adata.obsm['tm_pseudo_expression'].columns), 
                            use_rep='tm_pseudo_expression_val')

        if self.spagft__method == 'spectral':
            self.adata.obs[self.cluster_key] = pd.Categorical(spectral_clustering(self.adata.obsp['connectivities'], n_clusters=self.spagft__n_clusters))
        else:
            sc.tl.louvain(self.adata, key_added=self.cluster_key)

        logging.info(r"SpaGFT clustering done. Added results to adata.obs['spagft']")

        
    def save_results(self):
        self.adata.uns['freq_signal_tm'] = [] # TODO enable DataFrames to be written to .h5ad. For now exclude them
        self.adata.uns['freq_signal_subTM'] = []
        self.adata.uns['gft_umap_tm'] = []
        if self.method == 'all':
            self.filename = self.adata.uns['sample_name'] + "_all"
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad.')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    # sc.settings.verbosity = 3      
    # sc.settings.set_figure_params(dpi=300, facecolor='white')
    # parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    # parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    # parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False)
    # args = parser.parse_args()

    # if not args.file.endswith('.h5ad'):
    #     raise AttributeError(f"File '{args.file}' extension is not .h5ad")
    
    # adata = sc.read(args.file)
    # if not scipy.sparse.issparse(adata.X):
    #     adata.X = scipy.sparse.csr_matrix(adata.X)

    # file_out = os.path.join(args.out_path, os.path.basename(args.file))

    # clustering_spagft(adata, file_out)

