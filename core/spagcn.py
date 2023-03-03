import sys
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import stereo as st
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
sys.path.append("/home/ubuntu/SpaGCN") # SpaGCN_package
import SpaGCN_package.SpaGCN as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import scanpy as sc
import argparse
import time
start = time.perf_counter()

parser = argparse.ArgumentParser(description='SVG identification with SpaGCN')
parser.add_argument('-f', '--file', help='Input anndata file.', type=str, required=True)
parser.add_argument('--skip_domain_calculation', help='Performs only identification of SVGs. Domains available in adata.obs[\'pred\']', type=bool)
parser.add_argument('--max_num_clusters', help='Max number of clusters.', type=int, default=15)
args = parser.parse_args()
#'/goofys/stereoseq_dataset/SS200000135TL_D1_with_annotation.h5ad'
# /goofys/stereoseq_dataset/SS200000135TL_D1_with_annotation.h5ad'
# Mouse_brain_SS200000141TL_A4.h5ad'. 3D_spatial
#/goofys/stereoseq_dataset/Sliced_Mouse_embryo_E9.5_E1S1.MOSTA.h5ad'

base_in_fname = args.file.split('/')[-1].split('.')[0]
if args.file.endswith('.h5ad'):
    adata = sc.read(args.file)
elif args.file.endswith('.gef'):
    data = st.io.read_gef(file_path=args.file, bin_type='cell_bins')
    adata = st.io.stereo_to_anndata(data)

adata = sc.read(args.file)
if not args.skip_domain_calculation:
    #Set coordinates
    adata.obs["x_array"]=adata.obsm['spatial'][:, 0]
    adata.obs["y_array"]=adata.obsm['spatial'][:, 1]
    adata.obs["x_pixel"]=adata.obsm['spatial'][:, 0]
    adata.obs["y_pixel"]=adata.obsm['spatial'][:, 1]

    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()
    #Run SpaGCN
    adata.obs["pred"]= spg.detect_spatial_domains_ez_mode(adata, None, x_array, y_array, x_pixel, y_pixel, n_clusters=args.max_num_clusters, histology=False, s=1, b=49, p=0.5, r_seed=100, t_seed=100, n_seed=100)
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Refine domains (optional)
    #shape="hexagon" for Visium data, "square" for ST data.
    adata.obs["refined_pred"]=spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), x_array=x_array, y_array=y_array, shape="hexagon")
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    ax=spg.plot_spatial_domains_ez_mode(adata, domain_name="pred", x_name="x_pixel", y_name="y_pixel", plot_color=plot_color,size=150000/adata.shape[0], show=False, save=True,save_dir=base_in_fname + "_pred.png")
    ax=spg.plot_spatial_domains_ez_mode(adata, domain_name="refined_pred", x_name="x_pixel", y_name="y_pixel", plot_color=plot_color,size=150000/adata.shape[0], show=False, save=True,save_dir=base_in_fname + "_refined_pred.png")
    adata.write(f'{base_in_fname}_spagcn.h5ad', compression="gzip")
    end_domaining = time.perf_counter()
    print(f'Domain calculation took: {end_domaining - start} sec.')

domains = set(adata.obs["pred"].values)
print(f'Found domains: {domains}')
# Find SVGs
adata.X=(adata.X.A if issparse(adata.X) else adata.X)
adata.raw=adata
# sc.pp.log1p(adata)
#Set filtering criterials
min_in_group_fraction=0.8
min_in_out_group_ratio=1
min_fold_change=1.5
all_filtered_info = pd.DataFrame()
for target in domains:
    print(f'Processing domain {target}...')
    filtered_info=spg.detect_SVGs_ez_mode(adata, target=target, x_name="x_array", y_name="y_array", domain_name="pred", min_in_group_fraction=min_in_group_fraction, min_in_out_group_ratio=min_in_out_group_ratio, min_fold_change=min_fold_change)
    if len(filtered_info) > 0:
        # If zero genes found for the domain
        filtered_info.loc[:, 'domain'] = int(target)
    all_filtered_info = pd.concat([all_filtered_info, filtered_info]) if len(all_filtered_info) > 0 else filtered_info
    print(f'Found {len(filtered_info)} DEGs for domain {target}')

all_filtered_info.to_csv(base_in_fname + '_spagcn_svgs.csv', index=False)
end = time.perf_counter()
print(f'Total execution time: {end - start} sec.')