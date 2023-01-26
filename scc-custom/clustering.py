import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.sparse import csr_matrix
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import louvain


#------------------------------------------Kmeans-------------------------------------------------------------------
def weight_scaler(dataframe, scaler):
    for i in dataframe.columns:
     if i in ['u1', 'u2']:
       dataframe[i] = dataframe[i].apply(lambda x: x * scaler)
     else:
       dataframe[i] = dataframe[i].apply(lambda x: x * (1 - scaler))
    return dataframe
# Creating DF with spatial coordination
def extracting_coordinates(dataframe): #from multiindex dataframe
    coordinates = pd.DataFrame(dataframe.values, columns = ['x','y'])
    return coordinates
#Concatination of umaps and spatial coordination
def concatination(anndata, coord_df):
    sc.tl.umap(anndata)
    umap_comp = pd.DataFrame(anndata.obsm['X_umap'], columns = ['u1','u2'])
    frames = [umap_comp, coord_df]
    concatinated_df = pd.concat(frames, axis = 1)
    return concatinated_df
# Clustering (Kmeans) from concatinated dataframe
def cluster_with_kmeans_and_plot(concat_df, scaler):
    df_normalised = normalize(concat_df, axis = 0) #returns sparse matrix or numpy object
    df_normalised = pd.DataFrame(df_normalised, columns=['u1','u2','x','y'])
    scaled_dataframe = weight_scaler(df_normalised, scaler)
    num_cl = 11
    colors = cm.rainbow(np.linspace(0, 1, num_cl))
    kmeans = KMeans(n_clusters=num_cl, init = 'k-means++', random_state = 42).fit(scaled_dataframe)
    label=kmeans.predict(scaled_dataframe)
    for i in range(0,num_cl):
        plt.scatter(scaled_dataframe.loc[label == i, 'x'], scaled_dataframe.loc[label == i, 'y'], color=colors[i], s = 1, marker='.')
    plt.title('Kmeans clustering (umap costum)')
    plt.savefig('Kmeans clustering (umap costum).png')

#------------------------Scanpy----------------------------------------------------------
def anndata_add_spatial(spat_coord, anndata):
    spat_coord = np.array(spat_coord) 
    anndata.obsm['spatial'] = spat_coord
    return anndata

def data_preprocesing(anndata):
    print('----------------> Normalisation of the data')
    sc.pp.normalize_total(anndata, target_sum=1e4, inplace = True)
    sc.pp.log1p(anndata)
    print('----------------> Calcualting PCA')
    sc.tl.pca(anndata, svd_solver='arpack')

def louv_clustering_umap(anndata):
    # for scnpy_umap
    sc.tl.umap(anndata)
    sc.tl.louvain(anndata, resolution = 1.0, key_added='louvain_1.0')
    sc.pl.spatial(anndata,
              color= 'louvain_1.0',
              title = 'Louvain clustering',
              gene_symbols=gene_id,
              spot_size = 25,
              save = 'Louvain clustering_1.0')
    print('----------------> Figure saved in current directory')

def louv_clustering_scc(anndata):
    sc.tl.louvain(anndata, 
              adjacency = (anndata.obsp['Gene expression_connectivities'] + anndata.obsp['physical_space_connectivities']),
              resolution = 1.8,
              key_added = 'scc_louvian_1.8'
    ) 
    sc.pl.spatial(anndata, color='scc_louvian_1.8', gene_symbols=gene_id, spot_size= 42, save = 'scc_louvain clustering_1.8')
    print('----------------> Figure saved in current directory')

def umap_simple_kmeans(adata, cell_coord, scaler):
    cell_coord_df = extracting_coordinates(cell_coord)
    data_preprocesing(adata)
    sc.pp.neighbors(adata, n_pcs=30)
    concat_dataframe = concatination(adata, cell_coord_df)
    cluster_with_kmeans_and_plot(concat_dataframe, scaler)
    print('----------------> Figure saved in current directory')
    
def umap_simple_louvain(adata, cell_coord):
    adata = anndata_add_spatial(cell_coord, adata)
    data_preprocesing(adata)
    sc.pp.neighbors(adata, n_pcs=30)
    louv_clustering_umap(adata)

def scc(adata, cell_coord):
    adata = anndata_add_spatial(cell_coord, adata)
    data_preprocesing(adata)
    # Creating neighborhood graph based on distance in transcriptomic space (30-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 30, n_pcs=30, key_added = 'Gene expression')
    #Creating neighborhood graph based on distance in physical space (8-nearest neighbors)
    sc.pp.neighbors(adata, n_neighbors = 8 , use_rep = 'spatial', key_added = 'physical_space') # key_added != 'spatial'
    louv_clustering_scc(adata)
#_____________________________________________________

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str)
    parser.add_argument('--weight', type = float)
    parser.add_argument('--method', choices = ['Umap simple Kmeans','Umap simple Louvain','SCC'])
    args = parser.parse_args()
    print(args.filename)

    sc.settings.set_figure_params(dpi=200)
    print('----------------> reading csv file')
    data = pd.read_csv(f'/mnt/c/Users/antic/Desktop/Test3/{args.filename}', compression='gzip', sep='\t')
    #--------------------------------dataset_transformation--------------------------------------------------
    print('----------------> Data tranformation')
    data.set_index('geneID', inplace = True)
    data_pivoted = pd.pivot_table(data, index = ['cell','x','y'], columns = data.index, aggfunc ='sum', fill_value = 0)
    cell_id = list(data_pivoted.index.get_level_values(0))
    cell_id = list(map(str, cell_id))
    gene_id = list(data_pivoted.columns.get_level_values(1))
    #-------------------- creating anndata object ------------------------------
    print('----------------> Creating AnnData object')
    sparse_matrix = scipy.sparse.csr_matrix(data_pivoted)
    adata = ad.AnnData(sparse_matrix)
    #---------------------------- assignnig cell and gene IDs -------------------------------
    df_pivot_reset_index = data_pivoted.reset_index(level =['cell','x','y'])
    cell_coord = df_pivot_reset_index.loc[:,['cell','x','y']]
    cell_coord = cell_coord.drop('cell', axis = 1)
    print('----------------> Assigning OBS and VAR names')
    adata.obs_names = cell_id
    adata.var_names = gene_id
    
    if args.method == 'Umap simple Kmeans':
        umap_simple_kmeans(adata, cell_coord, args.weight)
    elif args.method == 'Umap simple Louvain':
        umap_simple_louvain(adata, cell_coord)
    else:
        scc(adata, cell_coord)
