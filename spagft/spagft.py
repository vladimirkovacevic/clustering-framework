import argparse as ap
import logging
import os

import SpaGFT as spg
import numpy as np
import pandas as pd
import scanpy as sc
import stereo as st
import matplotlib.pyplot as plt


def load_and_preprocess(file):
    adata = sc.read(file)
    adata.var_names_make_unique()
    adata.raw = adata
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    logging.info(f'Successfully read the file: {file}')
    return adata


def identify_svg_and_tissue_modules(adata, spatial_key):
    if spatial_key in adata.obsm_keys() or set(spatial_key) <= set(adata.obs_keys()):
        spg.rank_gene_smooth(adata,
                            ratio_low_freq=0.5,
                            ratio_high_freq=3,
                            ratio_neighbors=1,
                            spatial_info=spatial_key)
        logging.info(f'Identified spatially variable genes')
        spg.gft.find_tissue_module(adata, 
                                    ratio_fms=2,
                                    ratio_neighbors=1,
                                    spatial_info=spatial_key,
                                    quantile=0.85)
        logging.info(f'Identified tissue modules')
        return True
    return False


def clustering(adata):
    adata.obsm['tm_pseudo_expression_val'] = adata.obsm['tm_pseudo_expression'].values
    sc.pp.neighbors(adata, 
                    n_neighbors=220, 
                    n_pcs=len(adata.obsm['tm_pseudo_expression'].columns), 
                    use_rep='tm_pseudo_expression_val')
    sc.tl.louvain(adata, key_added="clusters")
    logging.info(f'Finished louvain clustering with neigh graph from tissue modules instead of PCA')


def save_clustering_result(adata, file):
    sc.pl.spatial(adata, color=['clusters'], spot_size=1.2, show=False)
    plt.savefig(file+'.png', dpi=250, bbox_inches='tight')
    out_fname = file.rstrip('.h5ad') + '.out.h5ad'
    adata.uns['freq_signal_tm'] = [] # TODO enable DataFrames to be written to .h5ad. For now exclude them
    adata.uns['freq_signal_subTM'] = []
    adata.uns['gft_umap_tm'] = []
    adata.write(out_fname, compression="gzip")
    logging.info(f'Saved clustering result as .png and {out_fname}.')


def main():
    logging.basicConfig(level=logging.INFO)
    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False)
    args = parser.parse_args()


    if not args.file.endswith('.h5ad'):
        raise AttributeError(f"File '{args.file}' extension is not .h5ad")
    adata = load_and_preprocess(args.file)
    if not (identify_svg_and_tissue_modules(adata, ['x', 'y']) or identify_svg_and_tissue_modules(adata, 'spatial')):
        raise KeyError("Spatial info is not avaliable in adata.obsm_keys == 'spatial' or adata.obs_keys ['x', 'y']")
    clustering(adata)
    
    save_clustering_result(adata, os.path.join(args.out_path, os.path.basename(args.file)))


if __name__ == '__main__':
    main()
