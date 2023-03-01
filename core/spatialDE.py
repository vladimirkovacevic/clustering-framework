import logging

import SpatialDE
import pandas as pd

from sklearn.cluster import spectral_clustering
from core import ClusteringAlgorithm
from .utils import timeit

class SpatialdeAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + "_spatialde"
        self.cluster_key = 'spatialde'

    @timeit
    def run(self):
        self.preprocess(normalize=False)
        df = self.adata.to_df()
        coords = pd.DataFrame(index=df.index, data=self.adata.obsm['spatial'], columns=['x','y'])
        results = SpatialDE.run(coords, df)
        self.adata.uns['svg'] = results

        if self.svg_only:
            logging.info(r"SpatialDE finished identifying spatially variable genes. Added results to adata.varm['svg']")
            return
        logging.info(r"SpatialDE finished identifying spatially variable genes. Added results to adata.varm['svg']")

        
    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')


