#!/usr/bin/env python
# coding: utf-8

# ## Analyse tissue

# In[1]:


import os
import json
import argparse
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import stereo as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from tqdm import tqdm


def generate_stats(adata, sample_name:str):
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

    report_dict['Sample'] = sample_name
    report_dict['Number of observations (cell-bins)'] = nobs
    report_dict['Number of genes'] = ngenes
    report_dict['Non-zero gene expressions in observations %'] = nzero_exps
    report_dict['Expressed genes per cell (mean)'] = meanval
    report_dict['Expressed genes per cell (std)'] = std

    sns.histplot(adata.obs['genes_expressed_per_cell'])
    plt.axvline(x=meanval, color='red')
    plt.title('{} - {}x{} obs x genes\nNumber of genes expressed in each cell. Mean = {}, std = {}'.format(fname.split('.')[0], nobs, ngenes, meanval, std))
    plt.savefig('genes_per_cell_{}'.format(sample_name), dpi=200, bbox_inches='tight')

    n_neighbors = 4
    sc.pp.neighbors(adata, n_neighbors=n_neighbors + 1,use_rep='spatial', knn=True, key_added='stats')

    cell_genes_expressed_dict = dict()
    non_zero_pos = adata.X.nonzero()  # [[0, 0, 0, 1, 1, 1, 2, 2], [5, 7, 8, 5, 7, 9, 6, 9]]#
    start_ind = 0
    for ind, (cell, gene) in tqdm(enumerate(zip(non_zero_pos[0], non_zero_pos[1]))):
        
        if (cell != non_zero_pos[0][start_ind]) or (ind == len(non_zero_pos[0]) - 1):
            end_ind = ind + 1 if (ind == len(non_zero_pos[0]) - 1) else ind
            cell_genes_expressed_dict[non_zero_pos[0][start_ind]] = non_zero_pos[1][start_ind: end_ind]  # adata.obs.index[cell]
            start_ind = ind
            

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


    # Percentage of expressed genes intersecting with neighboring cells
    meanval = np.round(cell_intersect_neigh_perc.mean(), 2)
    std = np.round(cell_intersect_neigh_perc.std(), 3)
    sns.histplot(cell_intersect_neigh_perc)
    plt.axvline(x=meanval, color='red')
    plt.title(f'{sample_name} - {nobs}x{ngenes} obs x genes\nDistribution of percentage of intersecting expressed \n genes with {n_neighbors} spatially closest observations \nMean = {meanval}, std = {std}')
    plt.savefig(f'dist_percentage_{sample_name}.png', dpi=200, bbox_inches='tight')

    report_dict[f'Mean of percentage of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(100*cell_intersect_neigh_perc.mean(), 2)
    report_dict[f'Std of percentage of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(100*cell_intersect_neigh_perc.std(), 2)

    # number of genes expressed per cell
    meanval = np.round(cell_intersect_neigh_count.mean(), 2)
    std = np.round(cell_intersect_neigh_count.std(), 3)
    fname0 = fname.split('.')[0]
    sns.histplot(cell_intersect_neigh_count)
    plt.axvline(x=meanval, color='red')
    plt.title(f'{sample_name} - {nobs}x{ngenes} obs x genes\nDistribution of number of intersecting expressed \n genes with {n_neighbors} spatially closest observations \nMean = {meanval}, std = {std}')
    plt.savefig(f'dist_count_{sample_name}.png', dpi=200, bbox_inches='tight')
    report_dict[f'Mean of number of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(cell_intersect_neigh_count.mean(), 2)
    report_dict[f'Std of number of intersecting expressed genes with {n_neighbors} spatially closest observations'] = np.round(cell_intersect_neigh_count.std(), 2)

    fig = sns.jointplot(x=cell_intersect_neigh_count, y=cell_intersect_neigh_perc, s=4)
    plt.xlabel('Number of intersecting genes', fontsize=5)
    plt.ylabel('Percentage of intersecting genes', fontsize=5)
    plt.title('{} - {}x{} obs x genes\nPercentage and number of intersecting genes \nwith {} closest neighboring cells'.format(
        sample_name.split('.')[0], nobs, ngenes, n_neighbors), fontsize=12, y=1.3, x=-2.5)
    plt.savefig(f'perc_number_{sample_name}.png', dpi=200, bbox_inches='tight')

    json.dump(report_dict, open(sample_name + '.json', 'w'), indent=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str, required=True)
    args = parser.parse_args()
    fname = args.fname. #  "/home/ubuntu/stereo-seq/SS200000135TL_D1_with_annotation.h5ad"#

    if fname.endswith('.h5ad'):
        adata = sc.read(os.path.join(path_local, fname))
    elif fname.endswith('.gef'):
        data = st.io.read_gef(file_path=f'{path_local}{fname}', bin_type='cell_bins')
        adata = st.io.stereo_to_anndata(data)
    else:
        print('Input format not supported')

    generate_stats(adata, fname.split('/')[-1])

