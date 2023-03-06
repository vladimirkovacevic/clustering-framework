import logging
import copy

import hotspot
import pandas as pd
import scanpy as sc

from sklearn.cluster import spectral_clustering
from core import ClusteringAlgorithm
from .utils import timeit

class HotspotAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_hotspot_hvg{not self.hotspot__use_full_gene_set}_hvgnt{self.hotspot__n_hvgs}_ur{not self.hotspot__use_normalized_data}_nm{self.hotspot__null_model}_nn{self.hotspot__n_neighbors}_fdrt{self.hotspot__fdr_threshold}__mgt{self.hotspot__min_gene_threshold}_nj{self.n_jobs}"
        
        self.cluster_key = 'hotspot'

    def preprocess(
        self,
        min_genes=200,
        min_cells=3,
        target_sum=1e4,
        normalize=True,
        use_hvgs = None,
        n_hvgs = None,
        use_raw = None
        ):
        self.adata.var_names_make_unique()
        sc.pp.filter_cells(self.adata, min_genes)
        sc.pp.filter_genes(self.adata, min_cells)
        # save raw counts
        if use_raw:
            self.adata.raw = self.adata
        if normalize:
            sc.pp.normalize_total(self.adata, target_sum, inplace=True)
        if "log1p" not in self.adata.uns_keys():
            sc.pp.log1p(self.adata)
        if use_hvgs:
            # calculate highly variable genes
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs)
            # keep only HVGs, remove the rest of genes
            # check use_raw label if the HVGs should have raw count or log-normalized
            self.adata = self.adata.raw.to_adata()[:, self.adata.var.highly_variable] if use_raw else self.adata[:, self.adata.var.highly_variable]
        logging.info(f'Finished preprocessing')

    @timeit
    def run(self):
        # preprocess data, extract highly varable genes if the flag is set
        self.preprocess(use_hvgs=(not self.hotspot__use_full_gene_set), n_hvgs=self.hotspot__n_hvgs, use_raw=(not self.hotspot__use_normalized_data))
        
        # hotspot pipeline
        logging.info(r"Starting hotspot pipeline")
        # hotspot is initialized with cell-cell similarity defined through 'spatial' key
        hs = hotspot.Hotspot(adata=self.adata, layer_key=None, model=self.hotspot__null_model,
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
        self.adata.uns['hotspot_svg'] = hs_genes.values
        logging.info(r"Hotspot finished identifying spatially variable genes. Added results to adata.uns['hotspot_svg']")

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
        self.adata.var['hotspot_svg_modules'] = hs.modules
        logging.info(r"Hotspot finished identifying spatially variable gene modules. Added results to adata.var['hotspot_svg_modules']")

        # spatial information on modules can be obtained through per-cell module score
        # This is useful for visualizing the general pattern of expression for genes in a module.
        # The output is a pandas DataFrame (cells x modules) and is also saved in hs.module_scores
        _ = hs.calculate_module_scores()
        if hs.module_scores.shape[1] == 0:
            logging.error(r'Zero modules created. Please decrease hotspot__min_gene_threshold.')

        # adding module scores as embedding for X
        self.adata.obsm['hotsot_embedding'] = hs.module_scores.values
        logging.info(r"Module scores saved in self.adata.obsm['hotspot_embedding']")

        if self.svg_only:
            return

        # [NOTE] add clustering based on module scores embeddings

    def save_results(self):
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad')

        pd.DataFrame(self.adata.uns['hotspot_svg']).to_csv(f'{self.filename}_svg.csv', index=True)
        self.adata.var['hotspot_svg_modules'].to_csv(f'{self.filename}_svg_modules.csv', index=True)


