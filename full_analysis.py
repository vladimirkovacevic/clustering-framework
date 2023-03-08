import argparse as ap
import logging
import os

import scipy
import scanpy as sc
# import stereo as st
from core import SccAlgo
# from core import SpatialdeAlgo
from core import HotspotAlgo
# from core import SpagftAlgo
# from core import SpagcnAlgo
from core import StamarkerAlgo


logging.basicConfig(level=logging.INFO)


def RunAnalysis(algo):
    algo.run()
    if algo.svg_only:
        algo.save_results()
        return
    if any(set(['celltype_pred', 'annotation']).intersection(set(algo.adata.obs_keys()))):
        algo.calculate_clustering_metrics()
        algo.plot_clustering_against_ground_truth()
        # algo.plot_tissue_domains_against_ground_truth()
    else:
        algo.plot_clustering(color=[algo.cluster_key], sample_name=f'{algo.filename}.png')

if __name__ == '__main__':

    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs SVG and tissue domain identification.')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-m', '--method', help='A type of tissue clustering method to perform', type=str, required=False, choices=['spagft', 'spatialde', 'scc', 'spagcn', 'hotspot', 'stamarker', 'all'], default='spagft')
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-r', '--resolution', help='All: Resolution of the clustering algorithm', type=float, required=False, default=2)
    parser.add_argument('--n_neigh_gene', help='SCC: Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    parser.add_argument('--n_neigh_space', help='SCC: Number of neighbors using spatial distance', type=float, required=False, default=8)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)
    parser.add_argument('--n_marker_genes', help='Number of marker genes used for tissue domain identification by intersection. Consider all genes by default.', type=int, required=False, default=-1)
    parser.add_argument('-v', '--verbose', help='Show logging messages', action='count', default=0)
    parser.add_argument('--n_jobs', help='Number of CPU cores for parallel execution', type=int, required=False, default=8)

    parser.add_argument('--svg_only', help='Perform only identification of spatially variable genes', type=bool)

    parser.add_argument('--spagft__method', help='Algorithm to be used after SpaGFT dim red', type=str, required=False, default='louvain', choices=['louvain','spectral'])
    parser.add_argument('--spagft__ratio_low_freq', help='ratio_low_freq', type=float, required=False, default=0.5)
    parser.add_argument('--spagft__ratio_high_freq', help='ratio_high_freq', type=float, required=False, default=3.5)
    parser.add_argument('--spagft__ratio_neighbors', help='ratio_neighbors', type=float, required=False, default=1)
    parser.add_argument('--spagft__ratio_fms', help='ratio_fms', type=float, required=False, default=2)
    parser.add_argument('--spagft__quantile', help='quantile', type=float, required=False, default=0.85)
    parser.add_argument('--spagft__n_neighbors', help='n_neighbors', type=float, required=False, default=20)
    parser.add_argument('--spagft__n_clusters', help='n_clusters', type=float, required=False, default=12)

    parser.add_argument('--hotspot__use_full_gene_set', help='HotspotAlgo: True - use whole gene set; False - use only highly variable genes for downstream analysis', type=bool, required=False, default=False)
    parser.add_argument('--hotspot__n_hvgs', help='HotspotAlgo: Number of highly variable genes used for downstream analysis', type=int, required=False, default=3000)
    parser.add_argument('--hotspot__use_normalized_data', help='HotspotAlgo: True - use log-normalized data; False - use raw data (gene/cell filtered, but not log-normalized) for downstream analysis', type=bool, required=False, default=False)
    parser.add_argument('--hotspot__null_model', help='HotspotAlgo: Null model of cell gene expression', type=str, required=False, default='danb', choices=['danb','bernoulli', 'normal', 'none'])
    parser.add_argument('--hotspot__n_neighbors', help='HotspotAlgo: Number of neighbors for KNN graph', type=int, required=False, default=30)
    parser.add_argument('--hotspot__core_only', help='HotspotAlgo: True: Ambiguous genes are not assined to modules (labeled -1); False: all genes are assigned to modules', type=bool, required=False, default=False)
    parser.add_argument('--hotspot__fdr_threshold', help='HotspotAlgo: FDR threshold for selection of genes with higher autocorrelation', type=float, required=False, default=0.05)
    parser.add_argument('--hotspot__min_gene_threshold', help='HotspotAlgo: Minimum number of genes per module', type=int, required=False, default=10)

    parser.add_argument('--spagcn__refine', help='refine clustering', type=bool, required=False)
    parser.add_argument('--spagcn__skip_domain_calculation', help='Skip domain calculation', type=bool, required=False)
    parser.add_argument('--spagcn__max_num_clusters', help='Max number of clusters', type=int, required=False, default=15)

    
    parser.add_argument('--stamarker__min_cells', help='Preprocessing - minimum number of cells in which gene express', type=int, required=False, default=50)
    parser.add_argument('--stamarker__min_counts', help='Preprocessing - minimum number of counts in cell', type=int, required=False, default=None)
    parser.add_argument('--stamarker__n_top_genes', help='Preprocessing - number of HVGs', type=int, required=False, default=3000)
    parser.add_argument('--stamarker__radial_cutoff', help='Preprocessing - radius cutoff for spatial neighbor network', type=float, required=False, default=50.0)
    parser.add_argument('--stamarker__n_auto_enc', help='StamarkerAlgo: Number of auto-encoders', type=int, required=False, default=20)
    parser.add_argument('--stamarker__clustering_method', help='StamarkerAlgo: clustering method',type=str, required=False, default='louvain', choices=['louvain','mclust'])
    parser.add_argument('--stamarker__n_clusters', help='StamarkerAlgo: wanted number of cluster (required for mclust and consensus clustering)', type=int, required=False, default=5)
    parser.add_argument('--stamarker__alpha', help='StamarkerAlgo: number of sigmas for SVG threshold', type=int, required=False, default=2)

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

    all_methods = {'scc':SccAlgo, 'hotspot':HotspotAlgo, 'stamarker':StamarkerAlgo} #'spagft':SpagftAlgo, 'spatialde':SpatialdeAlgo, 'hotspot':HotspotAlgo, 'spagcn': SpagcnAlgo, 'stamarker':StamarkerAlgo}
    if args.method == 'all':
        for method in all_methods:
            algo = all_methods[method](adata, **vars(args))
            RunAnalysis(algo)
    else:
        algo = all_methods[args.method](adata, **vars(args))
        RunAnalysis(algo)

    algo.save_results()