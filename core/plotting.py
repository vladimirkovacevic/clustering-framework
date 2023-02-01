import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, v_measure_score, mutual_info_score
import argparse as ap
import scipy


def plot_clustering_against_ground_truth(
    adata:sc.AnnData, 
    labels_true='sim anno',
    labels_true_colors='sim anno_colors', 
    labels_pred='scc',
    sample_name='E16.5_E1S3_cell_bin_whole_brain_spagft_1',
    spot_size=100):
    # Remap obtained clustering to ground truth
    cm = contingency_matrix(labels_true=adata.obs[labels_true], labels_pred=adata.obs[labels_pred])
    cont_df = pd.DataFrame(cm, index=sorted(set(adata.obs[labels_true])), columns=sorted(set(adata.obs[labels_pred])))
    remap_dict = {col:cont_df.index[np.argmax(cont_df[col])] for col in cont_df.columns}
    adata.obs.loc[:, labels_pred + '_remap'] = adata.obs[labels_pred].map(remap_dict)

    remaped_classes = set(adata.obs[labels_pred + '_remap'])
    mask_col = [1 if x in remaped_classes else 0 for x in cont_df.index]
    colors = list(adata.uns[labels_true + '_colors'])
    remaped_colors = []
    for m, c in zip(mask_col, colors):
        if m:
            remaped_colors.append(c)

    # Plot ground truth
    
    sc.pl.spatial(adata, color=[labels_true], palette=colors, spot_size=spot_size)
    plt.savefig(f'{sample_name}_ground_truth.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Plot remaped clusters
    sc.pl.spatial(adata, color=[labels_pred + '_remap'], palette=remaped_colors, spot_size=spot_size)
    plt.savefig(f'{sample_name}.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    sc.settings.verbosity = 3      
    sc.settings.set_figure_params(dpi=300, facecolor='white')
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Path to store outputs', type=str, required=False)
    # parser.add_argument('-s', '--spot_size', help='Size of the spot on plot', type=float, required=False, default=1.2)
    args = parser.parse_args()

    if not args.file.endswith('.h5ad'):
        raise AttributeError(f"File '{args.file}' extension is not .h5ad")
    
    adata = sc.read(args.file)
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    plot_clustering_against_ground_truth(adata)