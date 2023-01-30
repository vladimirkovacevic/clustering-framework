import scanpy as sc
import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def plot_clustering_against_ground_truth(
    adata:sc.AnnData, 
    labels_true='annotation',
    labels_true_colors='annotation_colors', 
    labels_pred='clusters',
    sample_name='Mouse_embryo',
    spot_size=1.5):
    # Remap obtained clustering to ground truth
    cm = contingency_matrix(labels_true=adata.obs[labels_true], labels_pred=adata.obs[labels_pred])
    cont_df = pd.DataFrame(cm, index=sorted(set(adata.obs[labels_true])), columns=sorted(set(adata.obs[labels_true])))
    remap_dict = {col:cont_df.index[np.argmax(cont_df[col])] for col in cont_df.columns}
    adata.obs.loc[:, labels_pred + '_remap'] = adata.obs[labels_pred].map(remap_dict)
    ari_score = adjusted_rand_score(labels_true=adata.obs[labels_true], labels_pred=adata.obs[labels_pred])

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
    plt.savefig(f'{sample_name}_ground_truth.png', dpi=200, bbox_inches='tight')
    plt.close()