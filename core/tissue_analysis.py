#!/usr/bin/env python
# coding: utf-8

# ## Analyse tissue

# In[70]:


import os
import json

import SpaGFT as spg
import numpy as np
import pandas as pd
import scanpy as sc
import stereo as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

sc.settings.verbosity = 3      
#sc.logging.print_header()
sc.settings.set_figure_params(dpi=200, facecolor='white')


# Define the folder to save the results:

# In[2]:


path_clustering = '/goofys/stereoseq_dataset/'
path_local = '/home/ubuntu/stereo-seq/'
file_list = [
    'Mouse_brain_SS200000141TL_A4.h5ad',
    'Mouse_embryo_E9.5_E1S1.MOSTA.h5ad',
    'SS200000135TL_D1_with_annotation.h5ad',
    'SS200000135TL_D1_with_annotation_low_version.h5ad'
]


# In[46]:


fname = file_list[1]
adata = sc.read(os.path.join(path_local, fname))
adata


# In[47]:


# Plot 1. Number of genes expressed in the count matrix , 
#      2. total counts per cell
#.     3. number of genes expressed per cell
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'genes_expressed_per_cell'],
#              jitter=0.4, multi_panel=True)


# In[56]:


fname0 = fname.split('.')[0]
report_dict = {}
nobs = adata.X.get_shape()[0] if sparse.issparse(adata.X) else adata.X.shape[0]
ngenes = adata.X.get_shape()[1] if sparse.issparse(adata.X) else adata.X.shape[1]
non_zero_count = adata.X.count_nonzero() if sparse.issparse(adata.X) else np.count_nonzero(adata.X)
adata.obs['genes_expressed_per_cell'] = np.diff(adata.X.tocsr().indptr) if sparse.issparse(adata.X) else (adata.X != 0).sum(1)
meanval = int(adata.obs['genes_expressed_per_cell'].mean())
std = np.round(adata.obs['genes_expressed_per_cell'].std(), 3)
print(f'Number of observations (cell-bins): {nobs}')
print(f'Number of genes: {ngenes}')
nzero_exps = np.round(100 * non_zero_count / (nobs * ngenes), 2)
print(f'Non-zero gene expressions in observations: {nzero_exps}%.')
print(f'Expressed genes per cell (mean, std): {meanval, std}')

report_dict['Sample'] = fname0
report_dict['Number of observations (cell-bins)'] = nobs
report_dict['Number of genes'] = ngenes
report_dict['Non-zero gene expressions in observations %'] = nzero_exps
report_dict['Expressed genes per cell (mean)'] = meanval
report_dict['Expressed genes per cell (std)'] = std


# In[49]:


# Calculate number of genes expressed per cell

sns.histplot(adata.obs['genes_expressed_per_cell'])
plt.axvline(x=meanval, color='red')
plt.title('{} - {}x{} obs x genes\nNumber of genes expressed in each cell. Mean = {}, std = {}'.format(fname.split('.')[0], nobs, ngenes, meanval, std))
plt.savefig('genes_per_cell_{}'.format(fname.split('.')[0]), dpi=200)


# In[50]:


n_neighbors = 4
sc.pp.neighbors(adata, n_neighbors=n_neighbors + 1,use_rep='spatial', knn=True, key_added='stats')


# In[73]:


cell_genes_expressed_dict = dict()
non_zero_pos = adata.X.nonzero()  # [[0, 0, 0, 1, 1, 1, 2, 2], [5, 7, 8, 5, 7, 9, 6, 9]]#
start_ind = 0
for ind, (cell, gene) in tqdm(enumerate(zip(non_zero_pos[0], non_zero_pos[1]))):
    
    if (cell != non_zero_pos[0][start_ind]) or (ind == len(non_zero_pos[0]) - 1):
        end_ind = ind + 1 if (ind == len(non_zero_pos[0]) - 1) else ind
        cell_genes_expressed_dict[non_zero_pos[0][start_ind]] = non_zero_pos[1][start_ind: end_ind]  # adata.obs.index[cell]
        start_ind = ind
        
#     if ind % 1000000 == 0:
#         print(f'{ind} out of {len(non_zero_pos[0])}')


# In[74]:


neighbors_in_graph = adata.obsp['stats_connectivities'].nonzero()
cell_intersect_neigh_perc = []
cell_intersect_neigh_count = []

for cell_id, curr_genes in tqdm(cell_genes_expressed_dict.items()):
    perc_intersects = []
    intersect_genes_counts = []
    for neigh in range(n_neighbors):
        curr_genes_set = set(cell_genes_expressed_dict[cell_id])
        neigh_index = neighbors_in_graph[1][cell_id * n_neighbors + neigh]
        neigh_genes_set = set(cell_genes_expressed_dict[neigh_index])
        intersect_genes_count = len(curr_genes_set.intersection(neigh_genes_set))
        perc_intersects.append(intersect_genes_count / len(curr_genes_set))
        intersect_genes_counts.append(intersect_genes_count)
    avg = np.mean(perc_intersects)
    cell_intersect_neigh_perc.append(avg)
    avg = np.mean(intersect_genes_counts)
    cell_intersect_neigh_count.append(avg)

cell_intersect_neigh_perc = np.array(cell_intersect_neigh_perc)
cell_intersect_neigh_count = np.array(cell_intersect_neigh_count)


# In[63]:


# number of genes expressed per cell
meanval = np.round(cell_intersect_neigh_perc.mean(), 2)
std = np.round(cell_intersect_neigh_perc.std(), 3)
sns.histplot(cell_intersect_neigh_perc)
plt.axvline(x=meanval, color='red')
plt.title(f'{fname0} - {nobs}x{ngenes} obs x genes\nDistribution of percentage of intersecting expressed \n genes with {n_neighbors} spatially closest observations \nMean = {meanval}, std = {std}')
plt.savefig(f'dist_percentage_{fname0}', dpi=200, bbox_inches='tight')

report_dict[f'Mean of percentage of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(100*cell_intersect_neigh_perc.mean(), 2)
report_dict[f'Std of percentage of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(100*cell_intersect_neigh_perc.std(), 2)


# In[62]:


# number of genes expressed per cell
meanval = np.round(cell_intersect_neigh_count.mean(), 2)
std = np.round(cell_intersect_neigh_count.std(), 3)
fname0 = fname.split('.')[0]
sns.histplot(cell_intersect_neigh_count)
plt.axvline(x=meanval, color='red')
plt.title(f'{fname0} - {nobs}x{ngenes} obs x genes\nDistribution of number of intersecting expressed \n genes with {n_neighbors} spatially closest observations \nMean = {meanval}, std = {std}')
plt.savefig(f'dist_count_{fname0}', dpi=200, bbox_inches='tight')
report_dict[f'Mean of number of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(cell_intersect_neigh_count.mean(), 2)
report_dict[f'Std of number of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(cell_intersect_neigh_count.std(), 2)


# In[55]:


fig = sns.scatterplot(x=cell_intersect_neigh_count, y=cell_intersect_neigh_perc, s=4)
plt.xlabel('Number of intersecting genes', fontsize=5)
plt.ylabel('Percentage of intersecting genes', fontsize=5)
plt.title('{} - {}x{} obs x genes\nPercentage and number of intersecting genes \nwith {} closest neighboring cells'.format(fname.split('.')[0], nobs, ngenes, n_neighbors), fontsize=13)
plt.savefig(f'perc_number_{fname0}', dpi=200, bbox_inches='tight')


# In[69]:


json.dump(report_dict, open(fname0 + '.json', 'w'), indent=True)
