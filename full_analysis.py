import argparse as ap
import logging
import os

import scipy
import scanpy as sc

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
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=70)
    args = parser.parse_args()

    if not args.file.endswith('.h5ad'):
        raise AttributeError(f"File '{args.file}' extension is not .h5ad")
    
    adata = sc.read(args.file)
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adataSpa = adata.copy()

    scc_algo = SccAlgo(adata, **vars(args))
    spagft_algo = SpagftAlgo(adataSpa, **vars(args))

    # ClusteringAlgorithm.calculate_clustering_metrics(scc_algo.adata)

    scc_algo.run()
    if 'annotation' in scc_algo.adata.obs:
        scc_algo.calculate_clustering_metrics()
        scc_algo.plot_clustering_against_ground_truth()
    else:
        scc_algo.plot_clustering()
    scc_algo.save_results()

    spagft_algo.run()
    if 'annotation' in spagft_algo.adata.obs:
        spagft_algo.calculate_clustering_metrics()
        spagft_algo.plot_clustering_against_ground_truth()
    else:
        spagft_algo.plot_clustering()
    spagft_algo.save_results()



    


