
import os

import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse


sc.settings.verbosity = 3      

path_local = '/home/ubuntu/stereo-seq/'
file_list = [
    'Mouse_brain_SS200000141TL_A4.h5ad',
    'Mouse_embryo_E9.5_E1S1.MOSTA.h5ad',
    'SS200000135TL_D1_with_annotation.h5ad',
    'SS200000135TL_D1_with_annotation_low_version.h5ad'
]

# adata = sc.read(f'{path_local}Mouse_embryo_E9.5_E1S1.MOSTA.h5ad')
fname = file_list[3]
adata = sc.read(os.path.join(path_local, fname))

nobs = adata.X.get_shape()[0] if sparse.issparse(adata.X) else adata.X.shape[0]
ngenes = adata.X.get_shape()[1] if sparse.issparse(adata.X) else adata.X.shape[1]
non_zero_count = adata.X.count_nonzero() if sparse.issparse(adata.X) else np.count_nonzero(adata.X)
print(f'Number of observations (cell-bins): {nobs}')
print(f'Number of genes: {ngenes}')
print(f'Non-zero gene expressions in observations: {np.round(100 * non_zero_count / (nobs * ngenes), 2)}%.')
print(f'Expressed genes per cell (mean, std): {meanval, std}')


# Calculate number of genes expressed per cell
adata.obs['genes_expressed_per_cell'] = np.diff(adata.X.tocsr().indptr) if sparse.issparse(adata.X) else (adata.X != 0).sum(1)
meanval = int(adata.obs['genes_expressed_per_cell'].mean())
std = np.round(adata.obs['genes_expressed_per_cell'].std(), 3)
sns.histplot(adata.obs['genes_expressed_per_cell'])
plt.axvline(x=meanval, color='red')
plt.title('{} - {}x{} obs x genes\nNumber of genes expressed in each cell. Mean = {}, std = {}'.format(fname.split('.')[0], nobs, ngenes, meanval, std))
plt.savefig('genes_per_cell_{}'.format(fname.split('.')[0]), dpi=200)

n_neighbors = 4
sc.pp.neighbors(adata, n_neighbors=n_neighbors + 1,use_rep='spatial', knn=True, key_added='stats')


cell_genes_expressed_dict = dict()
non_zero_pos = adata.X.nonzero()  # [[0, 0, 0, 1, 1, 1, 2, 2], [5, 7, 8, 5, 7, 9, 6, 9]]#
start_ind = 0
for ind, (cell, gene) in enumerate(zip(non_zero_pos[0], non_zero_pos[1])):
    
    if (cell != non_zero_pos[0][start_ind]) or (ind == len(non_zero_pos[0]) - 1):
        end_ind = ind + 1 if (ind == len(non_zero_pos[0]) - 1) else ind
        cell_genes_expressed_dict[non_zero_pos[0][start_ind]] = non_zero_pos[1][start_ind: end_ind]  # adata.obs.index[cell]
        start_ind = ind
        
    if ind % 1000000 == 0:
        print(f'{ind} out of {len(non_zero_pos[0])}')

neighbors_in_graph = adata.obsp['stats_connectivities'].nonzero()
cell_intersect_neigh_perc = []
for cell_id, curr_genes in cell_genes_expressed_dict.items():
    perc_intersects = []
    for neigh in range(n_neighbors):
        curr_genes_set = set(cell_genes_expressed_dict[cell_id])
        neigh_index = neighbors_in_graph[1][cell_id * n_neighbors + neigh]
        neigh_genes_set = set(cell_genes_expressed_dict[neigh_index])
        intersect_genes_count = len(curr_genes_set.intersection(neigh_genes_set))
        perc_intersects.append(intersect_genes_count / len(curr_genes_set))
    avg = np.mean(perc_intersects)
    cell_intersect_neigh_perc.append(avg)

    if cell_id % 10000 == 0:
        print(f'{cell_id} out of {len(cell_genes_expressed_dict.items())}')
cell_intersect_neigh_perc = np.array(cell_intersect_neigh_perc)      


# number of genes expressed per cell
meanval = np.round(cell_intersect_neigh_perc.mean(), 2)
std = np.round(cell_intersect_neigh_perc.std(), 3)
fname0 = fname.split('.')[0]
sns.histplot(cell_intersect_neigh_perc)
plt.axvline(x=meanval, color='red')
plt.title(f'{fname0} - {nobs}x{ngenes} obs x genes\nDistribution of percentage of intersecting expressed \n genes with {n_neighbors} spatially closest cells \nMean = {meanval}, std = {std}')
plt.savefig(f'dist_percentage_{fname0}', dpi=200, bbox_inches='tight')

