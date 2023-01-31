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

    def run(self):
        self.preprocess()
        spatial_key = 'spatial' if 'spatial' in adata.obsm_keys() else ['x', 'y'] if set(['x', 'y']) <= set(adata.obs_keys()) else None

        if not spatial_key:
            raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
        # find SVGs
        spg.rank_gene_smooth(adata,
                            ratio_low_freq=0.5,
                            ratio_high_freq=3,
                            ratio_neighbors=1,
                            spatial_info=spatial_key)
        logging.info(f'Identified spatially variable genes')
        # identify tissue modules
        spg.gft.find_tissue_module(adata, 
                                    ratio_fms=2,
                                    ratio_neighbors=1,
                                    spatial_info=spatial_key,
                                    quantile=0.85)
        logging.info(f'Identified tissue modules')

        adata.obsm['tm_pseudo_expression_val'] = adata.obsm['tm_pseudo_expression'].values
        sc.pp.neighbors(adata, 
                        n_neighbors=220, 
                        n_pcs=len(adata.obsm['tm_pseudo_expression'].columns), 
                        use_rep='tm_pseudo_expression_val')
        sc.tl.louvain(adata, key_added="spagft")
        logging.info(r"SpaGFT clustering done. Added results to adata.obsm['spagft']")

        
    def save_results(self):
        self.adata.uns['freq_signal_tm'] = [] # TODO enable DataFrames to be written to .h5ad. For now exclude them
        self.adata.uns['freq_signal_subTM'] = []
        self.adata.uns['gft_umap_tm'] = []
        self.adata.uns['sample_name'] = os.path.join(self.adata.uns['algo_params']['out_path'], os.path.basename(self.adata.uns['algo_params']['file'].rsplit(".", 1)[0]))
        filename = adata.uns['sample_name'] + f"_spagft_{self.resolution}.h5ad" 
        self.adata.write(filename, compression="gzip")
        logging.info(f'Saved clustering result {filename}.h5ad.')


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

