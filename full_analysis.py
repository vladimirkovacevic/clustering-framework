import argparse as ap
import logging
import os

import scipy
import scanpy as sc
import stereo as st

from core import calculate_clustering_metrics
from core import SpagftAlgo
from core import SccAlgo

logging.basicConfig(level=logging.INFO)

def PerformClustering(algo):
    algo.run()
    if any(set(['celltype_pred', 'annotation']).intersection(set(algo.adata.obs_keys()))):
        algo.calculate_clustering_metrics()
        algo.plot_clustering_against_ground_truth()
    else:
        algo.plot_clustering(sample_name=f'{algo.filename}.png')

if __name__ == '__main__':

    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-m', '--method', help='A type of tissue clustering method to perform', type=str, required=False, choices=['spagft', 'scc', 'all'], default='spagft')
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-r', '--resolution', help='All: Resolution of the clustering algorithm', type=float, required=False, default=2)
    parser.add_argument('--n_neigh_gene', help='SCC: Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    parser.add_argument('--n_neigh_space', help='SCC: Number of neighbors using spatial distance', type=float, required=False, default=8)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)
    parser.add_argument('-v', '--verbose', help='Show logging messages', action='count', default=0)

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

    all_methods = {'scc':SccAlgo, 'spagft':SpagftAlgo}
    if args.method == 'all':
        for method in all_methods:
            algo = all_methods[method](adata, **vars(args))
            PerformClustering(algo)
    else:
        algo = all_methods[args.method](adata, **vars(args))
        PerformClustering(algo)

    algo.save_results()