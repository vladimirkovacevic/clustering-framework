import logging
import scanpy as sc

from core import ClusteringAlgorithm
from .utils import timeit


class LeidenLouvainAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)

    @timeit
    def run(self):
        self.preprocess()
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neigh_gene, n_pcs=self.n_pcs)
        logging.info('Computing neighbors done')

        if self.cluster_key == 'leiden':
            sc.tl.leiden(
                self.adata,
                resolution=self.resolution,
                key_added = self.cluster_key
            )
        else:
            sc.tl.louvain(
                self.adata,
                resolution=self.resolution,
                key_added = self.cluster_key
            )
        logging.info(f"{self.cluster_key} clustering done. Added results to adata.obs[\'{self.cluster_key}\']")

    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')

    def set_method(self, cluster_key):
        self.cluster_key = cluster_key
        self.filename = self.adata.uns['sample_name'] + f"_{self.cluster_key}_ng{self.n_neigh_gene}_r{self.resolution}_pcs{self.n_pcs}"

