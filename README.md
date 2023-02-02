<div class="markdown prose w-full break-words dark:prose-invert light"><h1>Clustering Script</h1><p>This script performs clustering on single-cell RNA sequencing data using SpaGFT and SCC algorithms. It takes input file, output path, resolution, and other parameters and produces clustering results in form of plots, clustering metrics, and saved results.</p><h2>Usage</h2><pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class="">php</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-php-template"><span class="xml">python script.py -f <span class="hljs-tag">&lt;<span class="hljs-name">file.h5ad</span>&gt;</span> -o <span class="hljs-tag">&lt;<span class="hljs-name">out_path</span>&gt;</span> --resolution <span class="hljs-tag">&lt;<span class="hljs-name">resolution</span>&gt;</span> --n_neigh_gene <span class="hljs-tag">&lt;<span class="hljs-name">n_neigh_gene</span>&gt;</span> --n_neigh_space <span class="hljs-tag">&lt;<span class="hljs-name">n_neigh_space</span>&gt;</span> -s <span class="hljs-tag">&lt;<span class="hljs-name">spot_size</span>&gt;</span>
</span></code></div></div></pre><h2>Arguments</h2><ul><li><code>-f/--file</code>: Required argument, it specifies the input file containing the data to be clustered. The file should be in <code>.h5ad</code> format.</li><li><code>-o/--out_path</code>: Optional argument, it specifies the path to store outputs.</li><li><code>-r/--resolution</code>: Optional argument, it specifies the resolution of the clustering algorithm. The default value is 2.</li><li><code>--n_neigh_gene</code>: Optional argument, it specifies the number of neighbors using pca of gene expression for SCC algorithm. The default value is 30.</li><li><code>--n_neigh_space</code>: Optional argument, it specifies the number of neighbors using spatial distance for SCC algorithm. The default value is 8.</li><li><code>-s/--spot_size</code>: Optional argument, it specifies the size of the spot on the plot. The default value is 70.</li></ul><h2>Outputs</h2><ul><li>Clustering plots</li><li>Clustering metrics</li><li>Saved results in the specified output path.</li></ul><h2>Note</h2><p>The script requires the following libraries to be installed:</p><ul><li>argparse</li><li>logging</li><li>os</li><li>scipy</li><li>scanpy</li></ul><h2>Dependencies</h2><p>The script uses the following classes from the <code>core</code> module:</p><ul><li>SpagftAlgo</li><li>SccAlgo</li><li>calculate_clustering_metrics</li><li>plot_clustering_against_ground_truth</li></ul></div>
