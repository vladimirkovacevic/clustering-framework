import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.sparse import csr_matrix
from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import louvain

logging.basicConfig(level = logging.INFO, format='%(levelname)s:%(message)s')

#def read_gem(data):
def read_gem(data: pd.DataFrame) -> AnnData:
    """
    Read GEM file and return AnnData object.
    
    Parameters:
        data (pd.DataFrame): Input data frame containing gene expression information.
    
    Returns:
        AnnData: AnnData object containing sparse matrix representation of gene expression data, 
                cell IDs stored in `obs_names` attribute, gene IDs stored in `var_names` attribute 
                and spatial cell coordinates stored in 'spatial' attribute of the `obsm` attribute.
                
    """
    data.set_index('geneID', inplace = True)
    data_pivoted = pd.pivot_table(data, index = ['cell','x','y'], columns = data.index, aggfunc ='sum', fill_value = 0)
    cell_id = list(data_pivoted.index.get_level_values(0))
    cell_id = list(map(str, cell_id))
    gene_id = list(data_pivoted.columns.get_level_values(1))
    sparse_matrix = scipy.sparse.csr_matrix(data_pivoted)
    adata = AnnData(sparse_matrix, dtype = np.float32)
    df_pivot_reset_index = data_pivoted.reset_index(level =['cell','x','y'])
    cell_coord = df_pivot_reset_index.loc[:,['cell','x','y']]
    cell_coord = np.array(cell_coord.drop('cell', axis = 1, level = 0))
    adata.obs_names = cell_id
    adata.var_names = gene_id
    adata.obsm['spatial'] = cell_coord
    logging.info('adata with \'spatial\' key in the \'.obs\' is created')

    return adata

# Kmeans clustering
def kmeans_spa(adata: AnnData, expression_weight: float) -> None:
    """
    Perform KMeans clustering on the UMAP with spatial information 
    and store the results in the input AnnData object.

    Parameters:
    -   adata (AnnData): An anndata object (adata) containing the cell information and spatial coordination.
    -   expression_weight (float): The weight to be given to the gene expression in clustering.

    Returns: 
    -   None (modifies adata in-place)

    Side Effects:
    -   Adds 'umap_with_spatial' key in adata.obsm
    -   Adds 'kmeans_spa' key in adata.obs

    """
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    adata.obsm['umap_with_spatial'] = np.concatenate((adata.obsm['X_umap'], adata.obsm['spatial']),
                                                            axis = 1,
                                                            dtype = np.float32)
    logging.info('  \'umap_with_spatial\' key added in adata.obsm')
    adata.obsm['umap_with_spatial'] = normalize(adata.obsm['umap_with_spatial'], axis = 0)
    mapper = lambda arr: [arr[0] * expression_weight, arr[1] * expression_weight, arr[2] * (1-expression_weight), arr[3] * (1-expression_weight)]
    adata.obsm['umap_with_spatial']= np.array(list(map(mapper, adata.obsm['umap_with_spatial'])))
    num_cl = 11
    kmeans = KMeans(n_clusters = num_cl, init = 'k-means++', random_state = 42, n_init = 10).fit(adata.obsm['umap_with_spatial'])
    label = kmeans.predict(adata.obsm['umap_with_spatial'])
    adata.obs['kmeans_spa'] = pd.Categorical(label)
    logging.info('\'kmeans_spa\' key added in adata.obs')

# Louvain clustering 
def louvain_spa(adata: AnnData) -> None:
    """
    Perform Louvain clustering on UMAP reduced gene expression data and store the results in the input AnnData object.

    Parameters:
    -   adata (AnnData): An AnnData object containing cell information and spatial coordinates.

    Returns:
    -    None (modifies adata in-place)

    Side Effects:
    -   Adds 'louvain_spa' key in adata.obs
    """
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution = 1.4, key_added='louvain_spa')
    logging.info('\'louvain_spa\' key added in adata.obs')
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution = 1.4, key_added='louvain_spa')
    logging.info('\'louvain_spa\' key added in adata.obs')

# SCC with Louvain or Leiden
def scc(adata: AnnData, clustering_type: str) -> None:
    """
    Preform spatialy-constraind clustering (SCC).
    Constructs two k-nearest neighbor graphs 
    based on distance in transcriptomic space and in the physical space and build union graph.
    Perform Louvain/Leiden clustering on union graph. 

    Parameters:
    -   adata (AnnData): An AnnData object containing cell information and gene expression data.
    -   clustering_type (str): Louvain or Leidan clustering, runned after creating union graph.
    
    Returns:
    -   None (modifies adata in-place)

    Side Effects:
    -   adds 'louvain_spa'/'leiden_spa' key in adata.obs
    """
    # Creating neighborhood graph based on distance in transcriptomic space (30-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 30, n_pcs=30, key_added = 'gene_expression')
    #Creating neighborhood graph based on distance in physical space (8-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 8 , use_rep = 'spatial', key_added = 'physical_space')
    #Creating union graph
    union_graph = adata.obsp['gene_expression_connectivities'] + adata.obsp['physical_space_connectivities']
    if clustering_type == 'scc_louvain':
        sc.tl.louvain(adata, 
                  adjacency = union_graph,
                  resolution = 1.4,
                key_added = 'scc_louvain')
        logging.info('\'scc_louvain\' key added in adata.obs')
    else:
        sc.tl.leiden(adata, 
                adjacency = union_graph,
                key_added = 'scc_leiden' )
        logging.info('\'scc_leiden\' key added in adata.obs')
#_________________________________________________________________________________________________________
if __name__ == '__main__':

    description = '''This script performs pre-processing, clustering and 
    visualization of gene expression data using various clustering methods and spatial coordinates.'''
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('--path', type = str, help = 'path to the .tsv file', required = True)
    parser.add_argument('--method', choices = ['kmeans_spa','louvain_spa','scc_louvain', 'scc_leiden', 'all'], required = True)
    parser.add_argument('--plot', choices = ['yes', 'no'], default = 'yes')
    parser.add_argument('--weight', type = float, required = 'kmeans_spa' in sys.argv)
    parser.add_argument('--write_adata', choices = ['yes', 'no'], default = 'yes')
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
    
    sc.pp.normalize_total(adata, target_sum=1e4, inplace = True)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')

    # Clustering by choosen method
    if args.method == 'all':
        kmeans_spa(adata, args.weight)
        louvain_spa(adata)
        scc(adata, 'scc_louvain')
        scc(adata, 'scc_leiden')
    elif args.method == 'kmeans_spa':
        kmeans_spa(adata, args.weight)
    elif args.method == 'louvain_spa':
        louvain_spa(adata)     
    else:
        scc(adata, args.method)
    
    if args.write_adata == 'yes':
        adata.write(f'results_with_{args.method}.h5ad')
    if args.plot == 'yes' and args.method == 'all':
        sc.pl.spatial(adata,
                    color = ['kmeans_spa','louvain_spa','scc_louvain', 'scc_leiden'], 
                    spot_size = 30,
                    save = 'all_clustering_type.png')
    else:
        sc.pl.spatial(adata, color = args.method, spot_size = 30, save = f'{args.method}.png')