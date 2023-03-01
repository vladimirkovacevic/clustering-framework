import argparse as ap
import logging
import os

import scanpy as sc
import matplotlib.pyplot as plt

from core import ClusteringAlgorithm
from .utils import timeit

class SccAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_scc_ng{self.n_neigh_gene}_ns{self.n_neigh_space}_r{self.resolution}_mg{self.n_marker_genes}" 
        self.cluster_key = 'scc'

    @timeit
    def run(self):
        self.preprocess()
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neigh_gene, n_pcs=40)
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neigh_space , use_rep='spatial', key_added='physical_space')
        logging.info(f'Computing neighbors done')

        conn = self.adata.obsp["connectivities"].copy()
        conn.data[conn.data > 0] = 1
        adj = conn + self.adata.obsp["physical_space_connectivities"]
        adj.data[adj.data > 0] = 1

        sc.tl.leiden(self.adata, 
                    adjacency = adj,
                    resolution = self.resolution,
                    key_added = self.cluster_key
            )
        logging.info(r"SCC clustering done. Added results to adata.obs['scc']")
    
    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')


if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='A script that performs spatially constrained clustering (SCC)')
    # parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    # parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False, default='results')
    # parser.add_argument('-r', '--resolution', help='Resolution of the clustering algorithm', type=float, required=False, default=1)
    # parser.add_argument('--n_neigh_gene', help='Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    # parser.add_argument('--n_neigh_space', help='Number of neighbors using spatial distance', type=float, required=False, default=8)
    # parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=1.2)
    # args = parser.parse_args()

    # if not args.file.endswith('.h5ad'):
    #     raise AttributeError(f"File '{args.file}' extension is not .h5ad")
    
    # scc_clustering(args)
