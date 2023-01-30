from core import calculate_clustering_metrics
from core import plot_clustering_against_ground_truth
import scanpy as sc

adata = sc.read("/home/ubuntu/results/Mouse_embryo_E9.5_E1S1.MOSTA.scc.out.h5ad")


res = calculate_clustering_metrics(adata)

print(res)

plot_clustering_against_ground_truth(adata, sample_name="Mouse_embryo_E9.5_E1S1.MOSTA")


