import argparse as ap
import logging
import os

import scanpy as sc
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)

parser = ap.ArgumentParser(description='A script that performs spatially constrained clustering (SCC)')
parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False, default='results')
parser.add_argument('-r', '--resolution', help='Resolution of the clustering algorithm', type=float, required=False, default=1)
parser.add_argument('--n_neigh_gene', help='Number of neighbors using pca of gene expression', type=float, required=False, default=30)
parser.add_argument('--n_neigh_space', help='Number of neighbors using spatial distance', type=float, required=False, default=8)
parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=1.2)
args = parser.parse_args()

if not args.file.endswith('.h5ad'):
    raise AttributeError(f"File '{args.file}' extension is not .h5ad")

adata = sc.read(args.file)
logging.info(f'Successfully read the file: {args.file}')

adata.var_names_make_unique()
adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
logging.info(f'Preprocessing done')

sc.pp.neighbors(adata, n_neighbors=args.n_neigh_gene, n_pcs=40)
sc.pp.neighbors(adata, n_neighbors=args.n_neigh_space , use_rep='spatial', key_added='physical_space')
logging.info(f'Computing neighbors done')

conn = adata.obsp["connectivities"].copy()
conn.data[conn.data > 0] = 1
adj = conn + adata.obsp["physical_space_connectivities"]
adj.data[adj.data > 0] = 1

sc.tl.leiden(adata, 
              adjacency = adj,
              resolution = args.resolution,
              key_added = 'clusters'
    )
logging.info(f'Clustering done')

file_out_no_extension = os.path.join(args.out_path, os.path.basename(args.file.rstrip('.h5ad')))

sc.pl.spatial(adata, color='clusters', spot_size=args.spot_size, show = False)
plt.savefig(file_out_no_extension + '_SCC.png', dpi=250, bbox_inches='tight')
adata.write(file_out_no_extension + '.scc.out.h5ad', compression="gzip")

logging.info(f'Saved clustering image and anndata to {args.out_path}')
