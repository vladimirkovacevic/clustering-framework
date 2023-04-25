import logging
import scanpy as sc

from core import ClusteringAlgorithm
from .utils import timeit


class LeidenLouvainAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.cluster_key = self.method
        self.filename = self.adata.uns['sample_name'] + f"_{self.cluster_key}_ng{self.n_neigh_gene}_r{self.resolution}_pcs{self.n_pcs}"
        self.clustering = {'leiden' : sc.tl.leiden, 'louvain' : sc.tl.louvain}


    @timeit
    def run(self):
        self.preprocess()
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neigh_gene, n_pcs=self.n_pcs)
        logging.info('Computing neighbors done')

        self.clustering[self.cluster_key](
            self.adata,
            resolution=self.resolution,
            key_added = self.cluster_key
        )

        logging.info(f"{self.cluster_key} clustering done. Added results to adata.obs[\'{self.cluster_key}\']")


    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')

