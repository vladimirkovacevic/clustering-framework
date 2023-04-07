import argparse as ap
import logging
import os

import scipy
import scanpy as sc
import stereo as st

logging.basicConfig(level=logging.INFO)




if __name__ == '__main__':

    parser = ap.ArgumentParser(description='A script that compares marker genes of two samples and outputs a similarity matrix')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    # parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=30)
    parser.add_argument('--n_marker_genes', help='Number of marker genes used for tissue domain identification by intersection. Consider all genes by default.', type=int, required=False, default=-1)
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
