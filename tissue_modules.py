import argparse as ap
import logging
import os

import scipy
import scanpy as sc

from anndata import AnnData

NUM_MARKER_GENES = 10

def preprocessing(adata):
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    sc.pp.regress_out(adata, ['total_counts'])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

def identify_marker_genes(
    adata: AnnData,
    groupby: str = None
    ):
    # if not groupby: # TODO make it such that classes are used for clustering if obs doesnt contain the key
    #     possible_label = set(['celltype_pred','annotation','louvain','leiden','scc','spagft']).intersection(set(algo.adata.obs_keys()))
    #     if any(possible_label):
    #         groupby = list(possible_label)[0]
    #     else:
    #         logging.info("Leiden clustering will be performed in order to generate the key of the observations for finding the marker genes")
    #         groupby = 'leiden' 
    #         sc.tl.leiden(adata)
    if groupby == 'leiden':
        logging.info("Leiden clustering will be performed in order to generate the key of the observations for finding the marker genes")
        sc.tl.leiden(adata)
    sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added=f"{groupby}_marker_genes")

def remap_predicted_to_actual(
    predicted: str,
    actual: str
):
    tm_to_mg_actual_dict = {name: set(adata.uns[actual]['names'][:NUM_MARKER_GENES]) for name in adata.uns[actual]['names'].dtype.names}
    tm_to_mg_predicted_dict = {name: set(adata.uns[predicted]['names'][:NUM_MARKER_GENES]) for name in adata.uns[predicted]['names'].dtype.names}

    def intersectMax(a, b):
        #this is bad keep working on it 
        return max(list(map(a.intersect(b))), key=lambda x: x[1])
    
    remap_predicted_dict = {k: list(map(intersectMax, v, tm_to_mg_actual_dict.values())) for k, v in tm_to_mg_predicted_dict.items()}

    return remap_predicted_dict



    



if __name__ == '__main__':

    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-v', '--verbose', help='Show logging messages', action='count', default=0)

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

    preprocessing(adata)
    identify_marker_genes(adata)
    identify_marker_genes(adata, groupby='leiden')

    