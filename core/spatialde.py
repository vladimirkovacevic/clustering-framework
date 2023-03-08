import logging

import SpatialDE
import pandas as pd

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
        self.adata.uns['svg_' + self.cluster_key] = results

        if self.svg_only:
            logging.info(r"SpatialDE finished identifying spatially variable genes. Added results to adata.uns['svg']")
            return
        
    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        df = self.adata.uns['svg_' + self.cluster_key]
        df = df[df['qval'] < 0.05]
        df = df.assign(pvals_adj=df.qval)
        df = df.assign(genes=df.g)
        df.to_csv(f'{self.filename}_svgs.csv')
        logging.info(f'Saved result {self.filename}.h5ad')


