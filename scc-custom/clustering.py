import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.sparse import csr_matrix
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import louvain

logging.basicConfig(level = logging.INFO, format='%(levelname)s:%(message)s')
def read_gem(data):
    logging.info('Creating anndata from gem file')
    data.set_index('geneID', inplace = True)
    data_pivoted = pd.pivot_table(data, index = ['cell','x','y'], columns = data.index, aggfunc ='sum', fill_value = 0)
    cell_id = list(data_pivoted.index.get_level_values(0))
    cell_id = list(map(str, cell_id))
    gene_id = list(data_pivoted.columns.get_level_values(1))
    sparse_matrix = scipy.sparse.csr_matrix(data_pivoted)
    adata = ad.AnnData(sparse_matrix, dtype = np.float32)
    df_pivot_reset_index = data_pivoted.reset_index(level =['cell','x','y'])
    cell_coord = df_pivot_reset_index.loc[:,['cell','x','y']]
    cell_coord = np.array(cell_coord.drop('cell', axis = 1, level = 0))
    adata.obs_names = cell_id
    adata.var_names = gene_id
    adata.obsm['spatial'] = cell_coord
    return adata

# Kmeans clustering
def kmeans_spa(adata, scaler):
    logging.info('Kmeans clustering - preprocessing data')
    sc.pp.normalize_total(adata, target_sum=1e4, inplace = True)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    adata.obsm['umap_with_spatial'] = np.concatenate((adata.obsm['X_umap'], adata.obsm['spatial']),
                                                            axis = 1,
                                                            dtype = np.float32)
    adata.obsm['umap_with_spatial'] = normalize(adata.obsm['umap_with_spatial'], axis = 0) #returns sparse matrix or numpy object
    mapper = lambda arr: [arr[0] * scaler, arr[1] * scaler, arr[2] * (1-scaler), arr[3] * (1-scaler)]
    adata.obsm['umap_with_spatial']= np.array(list(map(mapper, adata.obsm['umap_with_spatial'])))
    num_cl = 11
    kmeans = KMeans(n_clusters = num_cl, init = 'k-means++', random_state = 42, n_init = 10).fit(adata.obsm['umap_with_spatial'])
    label = kmeans.predict(adata.obsm['umap_with_spatial'])
    adata.obs['kmeans_spa'] = label

# Louvain clustering 
def louvain_spa(adata):
    logging.info('Louvain clustering - preprocessing data')
    sc.pp.normalize_total(adata, target_sum=1e4, inplace = True)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution = 1.4, key_added='louvain_spa')

# SCC with Louvain or Leiden
def scc(adata, clustering_type):
    logging.info('SCC clustering - preprocessing data')
    sc.pp.normalize_total(adata, target_sum=1e4, inplace = True)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    logging.info('SCC clustering - creating neighborhood graph')
    # Creating neighborhood graph based on distance in transcriptomic space (30-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 30, n_pcs=30, key_added = 'gene_expression')
    #Creating neighborhood graph based on distance in physical space (8-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 8 , use_rep = 'spatial', key_added = 'physical_space') # key_added != 'spatial'
    if clustering_type == 'scc_louvain':
        sc.tl.louvain(adata, 
                  adjacency = (adata.obsp['gene_expression_connectivities'] + adata.obsp['physical_space_connectivities']),
                  resolution = 1.4,
                key_added = 'scc_louvain')
    else:
        sc.tl.leiden(adata, 
                adjacency = (adata.obsp['gene_expression_connectivities'] + adata.obsp['physical_space_connectivities']),
                key_added = 'scc_leiden' )
#_________________________________________________________________________________________________________
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Program for clustering' )
    parser.add_argument('--path', type = str, help = 'path to the .tsv file', required = True)
    parser.add_argument('--method', choices = ['kmeans_spa','louvain_spa','scc_louvain', 'scc_leiden'], required = True)
    parser.add_argument('--plot', choices=['yes', 'no'], required = True)
    parser.add_argument('--weight', type = float, required = 'kmeans_spa' in sys.argv)
    #parser.add_argument('--clust_type', choices = ['louvain', 'leiden'], required = 'SCC' in sys.argv)
    args = parser.parse_args()
    
    # Create anndata
    if args.path.endswith('.tsv'):
        data = pd.read_csv(args.path, sep='\t')
        adata = read_gem(data)
    elif args.path.endswith('.tsv.gz'):
        data = pd.read_csv(args.path, sep='\t', compression = 'gzip')
        adata = read_gem(data)
    elif args.path.endswith('.h5ad'):
        adata = sc.read_h5ad(args.path)
    
    # Clustering by choosen method
    if args.method == 'kmeans_spa':
        kmeans_spa(adata, args.weight)
    elif args.method == 'louvain_spa':
        louvain_spa(adata)     
    else:
        scc(adata, args.method)
    
    #adata.write(f'results_with_{args.method}.h5ad')
    if args.plot == 'yes':
        sc.pl.spatial(adata, color = args.method, spot_size = 30, save = f'{args.method}.png')