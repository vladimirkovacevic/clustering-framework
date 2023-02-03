import argparse as ap
import logging
import os

import scipy
import scanpy as sc
import stereo as st

from core import calculate_clustering_metrics
from core import plot_clustering_against_ground_truth
from core import SpagftAlgo
from core import SccAlgo
# from core import ClusteringAlgorithm



logging.basicConfig(level=logging.INFO)


# adata = sc.read("/home/ubuntu/results/Mouse_embryo_E9.5_E1S1.MOSTA.scc.out.h5ad")


# res = calculate_clustering_metrics(adata)

# print(res)

# plot_clustering_against_ground_truth(adata, sample_name="Mouse_embryo_E9.5_E1S1.MOSTA")


if __name__ == '__main__':

    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False)
    parser.add_argument('-r', '--resolution', help='All: Resolution of the clustering algorithm', type=float, required=False, default=2)
    parser.add_argument('--n_neigh_gene', help='SCC: Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    parser.add_argument('--n_neigh_space', help='SCC: Number of neighbors using spatial distance', type=float, required=False, default=8)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)

    parser.add_argument('--spagft__method', help='Algorithm to be used after SpaGFT dim red', type=str, required=False, default='louvain', choices=['louvain','spectral'])
    parser.add_argument('--spagft__ratio_low_freq', help='ratio_low_freq', type=float, required=False, default=0.5)
    parser.add_argument('--spagft__ratio_high_freq', help='ratio_high_freq', type=float, required=False, default=3.5)
    parser.add_argument('--spagft__ratio_neighbors', help='ratio_neighbors', type=float, required=False, default=1)
    parser.add_argument('--spagft__ratio_fms', help='ratio_fms', type=float, required=False, default=2)
    parser.add_argument('--spagft__quantile', help='quantile', type=float, required=False, default=0.85)
    parser.add_argument('--spagft__n_neighbors', help='n_neighbors', type=float, required=False, default=20)
    parser.add_argument('--spagft__n_clusters', help='n_clusters', type=float, required=False, default=12)

    args = parser.parse_args()

    if not (args.file.endswith('.h5ad') or args.file.endswith('.gef')):
        raise AttributeError(f"File '{args.file}' extension is not .h5ad or .gef")
    
    if args.file.endswith('.h5ad'):
        adata = sc.read(args.file)
    elif args.file.endswith('.gef'):
        data = st.io.read_gef(file_path=args.file, bin_type='cell_bins')
        adata = st.io.stereo_to_anndata(data)

    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adataSpa = adata.copy()

    # scc_algo = SccAlgo(adata, **vars(args))
    spagft_algo = SpagftAlgo(adataSpa, **vars(args))

    # ClusteringAlgorithm.calculate_clustering_metrics(scc_algo.adata) #TODO consider making this function static as well as plotting and analysis functions

    # scc_algo.run()
    # if any(set(['celltype_pred', 'annotation']).intersection(set(spagft_algo.adata.obs_keys()))):
    #     scc_algo.calculate_clustering_metrics()
    #     scc_algo.plot_clustering_against_ground_truth()
    # else:
    #     scc_algo.plot_clustering(sample_name=f'{scc_algo.filename}.png')
    # scc_algo.save_results()

    spagft_algo.run()
    if any(set(['celltype_pred', 'annotation']).intersection(set(spagft_algo.adata.obs_keys()))):
        spagft_algo.calculate_clustering_metrics()
        spagft_algo.plot_clustering_against_ground_truth()
    else:
        spagft_algo.plot_clustering(sample_name=f'{spagft_algo.filename}.png')
    spagft_algo.save_results()



    


