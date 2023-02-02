<h1>Clustering Script</h1>

<p>This script performs clustering on single-cell RNA sequencing data using SpaGFT and SCC algorithms. It takes input file, output path, resolution, and other parameters and produces clustering results in form of plots, clustering metrics, and saved results.</p>

<h2>Usage</h2>
<code>python script.py -f <file.h5ad> -o <out_path> --resolution <resolution> --n_neigh_gene <n_neigh_gene> --n_neigh_space <n_neigh_space> -s <spot_size>
</code>

<h2>Arguments</h2><ul><li><code>-f/--file</code>: Required argument, it specifies the input file containing the data to be clustered. The file should be in <code>.h5ad</code> format.</li><li><code>-o/--out_path</code>: Optional argument, it specifies the path to store outputs.</li><li><code>-r/--resolution</code>: Optional argument, it specifies the resolution of the clustering algorithm. The default value is 2.</li><li><code>--n_neigh_gene</code>: Optional argument, it specifies the number of neighbors using pca of gene expression for SCC algorithm. The default value is 30.</li><li><code>--n_neigh_space</code>: Optional argument, it specifies the number of neighbors using spatial distance for SCC algorithm. The default value is 8.</li><li><code>-s/--spot_size</code>: Optional argument, it specifies the size of the spot on the plot. The default value is 70.</li></ul><h2>Outputs</h2><ul><li>Clustering plots</li><li>Clustering metrics</li><li>Saved results in the specified output path.</li></ul>

<h2>Note</h2><p>The script requires the following libraries to be installed:</p><ul><li>argparse</li><li>logging</li><li>os</li><li>scipy</li><li>scanpy</li></ul>

<h2>Dependencies</h2><p>The script uses the following classes from the <code>core</code> module:</p><ul><li>SpagftAlgo</li><li>SccAlgo</li><li>calculate_clustering_metrics</li><li>plot_clustering_against_ground_truth</li></ul></div>
