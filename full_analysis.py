import argparse as ap
import logging
import os
import re

import scipy
import scanpy as sc
import stereo as st
from core import *



if __name__ == '__main__':

    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs SVG and tissue domain identification.')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-m', '--methods', help='Comma separated list of methods to perform. Available: spagft, spatialde, scc, leiden, louvain, spagcn, hotspot', type=str, required=True, default='spagft')
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-r', '--resolution', help='All: Resolution of the clustering algorithm', type=float, required=False, default=2)
    parser.add_argument('--n_neigh_gene', help='SCC: Number of neighbors using pca of gene expression', type=float, required=False, default=30)
    parser.add_argument('--n_neigh_space', help='SCC: Number of neighbors using spatial distance', type=float, required=False, default=8)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)
    parser.add_argument('--n_marker_genes', help='Number of marker genes used for tissue domain identification by intersection. Consider all genes by default.', type=int, required=False, default=-1)
    parser.add_argument('-v', '--verbose', help='Show logging messages. 0 - Show warrnings, >0 show info, <0 no output generated.', type=int, default=0)
    parser.add_argument('--n_jobs', help='Number of CPU cores for parallel execution', type=int, required=False, default=8)
    parser.add_argument('--svg_only', help='Perform only identification of spatially variable genes', action='store_true')
    parser.add_argument('--svg_cutoff', help='Cutoff pval adj for SVGs', type=float, default=0.05)

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
    args = parser.parse_args()

    if args.verbose == 0:
        logging.basicConfig(level=logging.WARNING, force=True)
    elif args.verbose > 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.NOTSET)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if args.file.endswith('.h5ad'):
        adata = sc.read(args.file)
    elif args.file.endswith('.gef'):
        data = st.io.read_gef(file_path=args.file, bin_type='cell_bins')
        adata = st.io.stereo_to_anndata(data)
    else:
        raise AttributeError(f"File '{args.file}' extension is not .h5ad or .gef")

    # Most algorithms demand sparse cell gene matrix
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # Parse requested and installed methods to make sure that requested methods are installed
    available_methods = [module.__name__ for module in sys.modules.values() if re.search('^core.+', module.__name__)]
    available_methods = [m.split('.')[1] for m in available_methods]

    chosen_methods = args.methods.split(',')
    assert set(chosen_methods).issubset(set(available_methods)), "The requested methods could not be executed because your environment lacks needed libraries."
        
    all_methods = {}
    if 'scc' in chosen_methods:
        all_methods['scc'] = SccAlgo
    if 'leiden' in chosen_methods:
        all_methods['leiden'] = LeidenAlgo
    if 'louvain' in chosen_methods:
        all_methods['louvain'] = LouvainAlgo
    if 'spagft' in chosen_methods:
        all_methods['spagft'] = SpagftAlgo
    if 'spatialde' in chosen_methods:
        all_methods['spatialde'] = SpatialdeAlgo
    if 'hotspot' in chosen_methods:
        all_methods['hotspot'] = HotspotAlgo
    if 'spagcn' in chosen_methods:
        all_methods['spagcn'] = SpagcnAlgo
    
    # Process requested methods
    for method in all_methods:
        algo = all_methods[method](adata, **vars(args))
        algo.run()
        if algo.svg_only:
            algo.save_results()
        else:
            if any(set(['celltype_pred', 'annotation']).intersection(set(algo.adata.obs_keys()))):
                algo.calculate_clustering_metrics()
                algo.plot_clustering_against_ground_truth()
            else:
                algo.plot_clustering()

    algo.save_results()

