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

from core import ClusteringAlgorithm

class SpagftAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_spagft_{self.resolution}"
        self.cluster_key = 'spagft'

    def run(self):
        self.preprocess()
        spatial_key = 'spatial' if 'spatial' in self.adata.obsm_keys() else ['x', 'y'] if set(['x', 'y']) <= set(self.adata.obs_keys()) else None

        if not spatial_key:
            raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
        # find SVGs
        spg.rank_gene_smooth(self.adata,
                            # ratio_low_freq=0.5,
                            # ratio_high_freq=3,
                            # ratio_neighbors=1,
                            spatial_info=spatial_key)
        logging.info(f'Identified spatially variable genes')
        # identify tissue modules
        spg.gft.find_tissue_module(self.adata, 
                                    # ratio_fms=2,
                                    # ratio_neighbors=1,
                                    spatial_info=spatial_key,
                                    # quantile=0.85,
                                    resolution=self.resolution
                                    )
        logging.info(f'Identified tissue modules')

        self.adata.obsm['tm_pseudo_expression_val'] = self.adata.obsm['tm_pseudo_expression'].values
        sc.pp.neighbors(self.adata, 
                        n_neighbors=220, 
                        n_pcs=len(self.adata.obsm['tm_pseudo_expression'].columns), 
                        use_rep='tm_pseudo_expression_val')
        sc.tl.louvain(self.adata, key_added=self.cluster_key)
        logging.info(r"SpaGFT clustering done. Added results to adata.obs['spagft']")

        
    def save_results(self):
        self.adata.uns['freq_signal_tm'] = [] # TODO enable DataFrames to be written to .h5ad. For now exclude them
        self.adata.uns['freq_signal_subTM'] = []
        self.adata.uns['gft_umap_tm'] = []
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

