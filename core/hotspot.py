import logging
import copy

import hotspot
import pandas as pd

from sklearn.cluster import spectral_clustering
from core import ClusteringAlgorithm
from .utils import timeit

class HotspotAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_hotspot_hvg{self.use_hvgs}_hvgnt{self.n_hvgs}_ur{self.use_raw}_nm{self.hotspot__null_model}_nn{self.hotspot__n_neighbors}_fdrt{self.hotspot__fdr_threshold}__mgt{self.hotspot__min_gene_threshold}_nj{self.n_jobs}"
        
        self.cluster_key = 'hotspot'

    @timeit
    def run(self):
        # preprocess data, calculated highly varable genes if the flag is set
        self.preprocess(normalize=False, use_hvgs=self.use_hvgs, n_hvgs=self.n_hvgs, use_raw=self.use_raw)
        
        # # check for if raw data is provided (this is done during self.preprocess())
        # if self.use_raw and not self.adata.raw:
        #     raise Exception(f'self.adata.raw must be set if use_raw is True.')
        
        # copy wanted data
        data = self.adata.raw.to_adata() if self.use_raw else self.adata
        # extract highly variable genes if needed
        data = data[:, self.adata.var.highly_variable] if self.use_hvgs else data

        # hotspot pipeline
        logging.info(r"Starting hotspot pipeline")
        # hotspot is initialized with cell-cell similarity defined through 'spatial' key
        hs = hotspot.Hotspot(adata=data, layer_key=None, model=self.hotspot__null_model,
                                latent_obsm_key='spatial', umi_counts_obs_key='total_counts')           
        logging.info(r"Hotspot object created")
        
        # create knn graph
        hs.create_knn_graph(weighted_graph=False, n_neighbors=self.hotspot__n_neighbors)
        logging.info(r"KNN graph created")
        
        # find informative genes by gene "autocorrelation"
        # fn returns pandas DataFrame with columns
        # C: Scaled -1:1 autocorrelation coeficients
        # Z: Z-score for autocorrelation
        # Pval: P-values computed from Z-scores
        # FDR: Q-values using the Benjamini-Hochberg procedure
        # The output is a pandas DataFrame (and is also saved in hs.results)
        _ = hs.compute_autocorrelations(jobs=self.n_jobs)
        # select genes with significant spatial autocorrelation
        hs_genes = hs.results.loc[hs.results.FDR < self.hotspot__fdr_threshold].index
        logging.info(f'Selected {len(hs_genes)} with FDR < {self.hotspot__fdr_threshold}')

        # save gene modules to adata.uns
        self.adata.uns['svg'] = hs_genes
        logging.info(r"Hotspot finished identifying spatially variable genes. Added results to adata.uns['svg']")

        # Compute pair-wise local correlations between selected genes
        # The output is a genes x genes pandas DataFrame of Z-scores 
        # for the local correlation values between genes. 
        # The output is also stored in hs.local_correlation_z.
        _ = hs.compute_local_correlations(genes=hs_genes, jobs=self.n_jobs)
        logging.info(r"Pair-wise local correlation of selected genes computed")

        # group genes into modules
        # The output is a pandas Series that maps gene to module number. 
        # Unassigned genes are indicated with a module number of -1. 
        # The output is also stored in hs.modules
        _ = hs.create_modules(min_gene_threshold=self.hotspot__min_gene_threshold,
                                core_only=True, fdr_threshold=self.hotspot__fdr_threshold)
        logging.info(f'Genes groupping into modules finished (min_gene_threshold={self.hotspot__min_gene_threshold}, fdr_threshold={self.hotspot__fdr_threshold}).')
        
        # save gene modules to adata.uns
        self.adata.uns['svg_modules'] = hs.modules
        logging.info(r"Hotspot finished identifying spatially variable gene modules. Added results to adata.uns['svg_modules']")

        # spatial information on modules can be obtained through per-cell module score
        # This is useful for visualizing the general pattern of expression for genes in a module.
        # The output is a pandas DataFrame (cells x modules) and is also saved in hs.module_scores
        _ = hs.calculate_module_scores()
        if hs.module_scores.shape[1] == 0:
            logging.error(r'Zero modules created. Please decrease hotspot__min_gene_threshold.')

        self.adata.obsm['embedding'] = hs.module_scores
        logging.info(r"Module scores saved in self.adata.obsm['embedding']")

        if self.svg_only:
            return

        # [NOTE] add clustering based on module scores embeddings

    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')

        self.adata.uns['svg'].to_csv(f'{self.filename}_svg.csv', index=True)
        self.adata.uns['svg_modules'].to_csv(f'{self.filename}_svg_modules.csv', index=True)


