import stereo as st
import pandas as pd
import scanpy as sc
import numpy as np
import pickle
import json

from stereo.core.stereo_exp_data import StereoExpData
from anndata import AnnData
from typing import Optional
from collections import defaultdict
from numpy import inf

ADJ_PVAL_CUTOFF = 0.05


def anndata_to_stereo(andata: AnnData, use_raw=False, spatial_key: Optional[str] = None):
    """
    transform the Anndata object into StereoExpData object.

    :param andata: input Anndata object,
    :param use_raw: use andata.raw.X if True else andata.X. Default is False.
    :param spatial_key: use .obsm[`'spatial_key'`] as position.
    :return: StereoExpData obj.
    """
    # data matrix,including X,raw,layer
    data = StereoExpData()
    data.exp_matrix = andata.layers['count'] if use_raw else andata.X
    # obs -> cell
    data.cells.cell_name = np.array(andata.obs_names)
    data.cells.n_genes_by_counts = andata.obs[
        'n_genes_by_counts'] if 'n_genes_by_counts' in andata.obs.columns.tolist() else None
    data.cells.total_counts = andata.obs['total_counts'] if 'total_counts' in andata.obs.columns.tolist() else None
    data.cells.pct_counts_mt = andata.obs['pct_counts_mt'] if 'pct_counts_mt' in andata.obs.columns.tolist() else None
    # var
    data.genes.gene_name = np.array(andata.var_names)
    data.genes.n_cells = andata.var['n_cells'] if 'n_cells' in andata.var.columns.tolist() else None
    data.genes.n_counts = andata.var['n_counts'] if 'n_counts' in andata.var.columns.tolist() else None
    # position
    data.position = andata.obsm[spatial_key] if spatial_key is not None else None
    return data

with open('annotation_to_marker.json', 'r') as handle:
    AnnoToMg = json.load(handle)
# sample = 'E13.5_E1S2'
sample = 'E10.5_E1S2'
adata = sc.read(f"/goofys/Samples/MOSTA/E10.5/{sample}.MOSTA.h5ad")

sdata = anndata_to_stereo(adata, spatial_key = 'spatial', use_raw=True)

sdata.tl.normalize_total(target_sum=10000)
sdata.tl.log1p()

df = pd.DataFrame({'bins': sdata.cell_names, 'group': adata.obs['annotation'].values})
sdata.tl.reset_key_record('cluster', 'annotation')
sdata.tl.result['annotation'] = df
sdata.tl.raw = sdata
sdata.tl.find_marker_genes(cluster_res_key='annotation', use_highly_genes=False, use_raw=False, sort_by='scores') #can be 'scores' or 'log2fc'
# sdata.tl.filter_marker_genes() #min_in_group_fraction=None, max_out_group_fraction=None, 

df_in = sdata.tl.result['marker_genes']['pct'].loc[:, sdata.tl.result['marker_genes']['pct'].columns != 'genes']
df2_out = sdata.tl.result['marker_genes']['pct_rest'].loc[:, sdata.tl.result['marker_genes']['pct_rest'].columns != 'genes']


marker_genes = defaultdict(list)
for k,v in sdata.tl.result['marker_genes_filtered'].items():
    if ".vs.rest" in k:
        v = v[v['pvalues_adj'] < ADJ_PVAL_CUTOFF]
        k = k[:-len(".vs.rest")]
        for i, gene in enumerate(v['genes']):
            if k in AnnoToMg.keys() and gene in AnnoToMg[k]:
                marker_genes[k].append((gene, i))

with open(f'stereo_markers_with_known_{sample}.json', 'w') as handle:
    json.dump(marker_genes, handle, 
                sort_keys=True,
                indent=4)
