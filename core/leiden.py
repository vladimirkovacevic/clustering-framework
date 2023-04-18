import logging
import scanpy as sc

from core import ClusteringAlgorithm
from .utils import timeit


class LeidenAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.cluster_key = 'leiden'
        self.filename = self.adata.uns['sample_name'] + f"_leiden_ng{self.n_neigh_gene}_r{self.resolution}_mg{self.n_marker_genes}"
    
    @timeit
    def run(self):
        self.preprocess()
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neigh_gene, n_pcs=40)
        logging.info('Computing neighbors done')

        sc.tl.leiden(
            self.adata,
            resolution=self.resolution,
            key_added = self.cluster_key
        )
        logging.info(r"Leiden clustering done. Added results to adata.obs['leiden']")

    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')

