import argparse as ap
import logging
import os
import sys

import SpaGFT as spg
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scipy
import stereo as st

from sklearn.cluster import spectral_clustering

if __name__ == '__main__':
    from clustering_algorithm import ClusteringAlgorithm
    from utils import timeit
else:
    from core import ClusteringAlgorithm
    from core.utils import timeit

logging.basicConfig(level=logging.INFO)

class SpagftAlgo(ClusteringAlgorithm):
    def __init__(self, adata, **params):
        super().__init__(adata, **params)
        self.filename = self.adata.uns['sample_name'] + f"_spagft_r{self.resolution}_rl{self.spagft__ratio_low_freq}_rh{self.spagft__ratio_high_freq}_rn{self.spagft__ratio_neighbors}__mg{self.n_marker_genes}"
        self.cluster_key = 'spagft'

    @timeit
    def run(self):
        if 'tm_pseudo_expression' not in self.adata.obsm_keys():
            self.preprocess()
            spatial_key = 'spatial' if 'spatial' in self.adata.obsm_keys() else ['x', 'y'] if set(['x', 'y']) <= set(self.adata.obs_keys()) else None

            if not spatial_key:
                raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
            # find SVGs
            gene_df = spg.rank_gene_smooth(self.adata,
                                ratio_low_freq=self.spagft__ratio_low_freq,
                                ratio_high_freq=self.spagft__ratio_high_freq,
                                ratio_neighbors=self.spagft__ratio_neighbors,
                                spatial_info=spatial_key)
            logging.info(f'Identified spatially variable genes')
            if self.svg_only:
                svg_list = gene_df[gene_df.cutoff_gft_score]\
                    [gene_df.qvalue < 0.05].index.tolist()
                gene_df = gene_df.loc[svg_list, :]
                self.adata.uns['svg_' + self.cluster_key] = gene_df
                return
            # identify tissue modules
            spg.gft.find_tissue_module(self.adata, 
                                        ratio_fms=self.spagft__ratio_fms,
                                        ratio_neighbors=self.spagft__ratio_neighbors,
                                        spatial_info=spatial_key,
                                        quantile=self.spagft__quantile,
                                        resolution=self.resolution
                                        )
            logging.info(f'Identified tissue modules')
            self.adata.obsm['tm_pseudo_expression_val'] = self.adata.obsm['tm_pseudo_expression'].values
            sc.pp.neighbors(self.adata, 
                            n_neighbors=self.spagft__n_neighbors, 
                            n_pcs=len(self.adata.obsm['tm_pseudo_expression'].columns), 
                            use_rep='tm_pseudo_expression_val')

        if self.spagft__method == 'spectral':
            self.adata.obs[self.cluster_key] = pd.Categorical(spectral_clustering(self.adata.obsp['connectivities'], n_clusters=self.spagft__n_clusters))
        else:
            sc.tl.louvain(self.adata, key_added=self.cluster_key)

        logging.info(r"SpaGFT clustering done. Added results to adata.obs['spagft']")

        
    def save_results(self):
        self.adata.uns['freq_signal_tm'] = [] # TODO enable DataFrames to be written to .h5ad. For now exclude them
        self.adata.uns['freq_signal_subTM'] = []
        self.adata.uns['gft_umap_tm'] = []
        if self.svg_only:
            self.adata.uns['svg_' + self.cluster_key].to_csv(f'{self.filename}_svgs.csv')
        if self.method == 'all':
            self.filename = self.adata.uns['sample_name'] + "_all"
        self.adata.write(f'{self.filename}.h5ad', compression="gzip")
        logging.info(f'Saved clustering result {self.filename}.h5ad.')


if __name__ == '__main__':

    parser = ap.ArgumentParser(description='A script that performs SVG and tissue domain identification.')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-m', '--method', help='A type of tissue clustering method to perform', type=str, required=False, choices=['spagft', 'spatialde', 'scc', 'spagcn', 'hotspot', 'all'], default='spagft')
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-r', '--resolution', help='All: Resolution of the clustering algorithm', type=float, required=False, default=2)
    parser.add_argument('--n_neigh_gene', help='SCC: Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    parser.add_argument('--n_neigh_space', help='SCC: Number of neighbors using spatial distance', type=float, required=False, default=8)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)
    parser.add_argument('--n_marker_genes', help='Number of marker genes used for tissue domain identification by intersection. Consider all genes by default.', type=int, required=False, default=-1)
    parser.add_argument('-v', '--verbose', help='Show logging messages', action='count', default=0)
    parser.add_argument('--svg_only', help='Perform only identification of spatially variable genes', action='store_true')

    parser.add_argument('--spagft__method', help='Algorithm to be used after SpaGFT dim red', type=str, required=False, default='louvain', choices=['louvain','spectral'])
    parser.add_argument('--spagft__ratio_low_freq', help='ratio_low_freq', type=float, required=False, default=0.5)
    parser.add_argument('--spagft__ratio_high_freq', help='ratio_high_freq', type=float, required=False, default=3.5)
    parser.add_argument('--spagft__ratio_neighbors', help='ratio_neighbors', type=float, required=False, default=1)
    parser.add_argument('--spagft__ratio_fms', help='ratio_fms', type=float, required=False, default=2)
    parser.add_argument('--spagft__quantile', help='quantile', type=float, required=False, default=0.85)
    parser.add_argument('--spagft__n_neighbors', help='n_neighbors', type=float, required=False, default=20)
    parser.add_argument('--spagft__n_clusters', help='n_clusters', type=float, required=False, default=12)

    args = parser.parse_args()

    if args.verbose == 0:
        logging.basicConfig(level=logging.WARNING, force=True)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not (args.file.endswith('.h5ad') or args.file.endswith('.gef')):
        raise AttributeError(f"File '{args.file}' extension is not .h5ad or .gef")
    
    if args.file.endswith('.h5ad'):
        adata = sc.read(args.file)
    elif args.file.endswith('.gef'):
        data = st.io.read_gef(file_path=args.file, bin_type='cell_bins')
        adata = st.io.stereo_to_anndata(data)

    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    algo = SpagftAlgo(adata, **vars(args))
    algo.run()
    if algo.svg_only:
        algo.save_results()
        sys.exit("Finished generating SVGs")
    if any(set(['celltype_pred', 'annotation']).intersection(set(algo.adata.obs_keys()))):
        algo.calculate_clustering_metrics()
        algo.plot_clustering_against_ground_truth()
        # algo.plot_tissue_domains_against_ground_truth()
    else:
        algo.plot_clustering(color=[algo.cluster_key], sample_name=f'{algo.filename}.png')

