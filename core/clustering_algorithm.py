import scanpy as sc
import logging

from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):

    def __init__(self, adata, **params):
        self.adata = adata
        for key, value in params.items():
            setattr(self, key, value)
        self.adata.uns['algo_params'] = params
    
    def preprocess(self):
        self.adata.var_names_make_unique()
        self.adata.raw = self.adata
        sc.pp.filter_genes(self.adata, min_cells=10)
        sc.pp.normalize_total(self.adata, inplace=True)
        sc.pp.log1p(self.adata)
        logging.info(f'Finished preprocessing')

    @abstractmethod
    def run():
        pass

    @abstractmethod
    def save_results(self):
        pass
