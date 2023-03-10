import argparse as ap
import os

import pandas as pd
import matplotlib.pyplot as plt

from upsetplot import from_contents, UpSet

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='A script that creates upsetplot for the identified SVG by different methods')
    parser.add_argument('-f', '--path', help='Absolute path to the folder that contains svg csv files', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    path = args.path
    out_path = args.out_path

    #csv files are expected to have columns genes and pvals_adj as well as being named Sample_METHOD_svgs.csv
    data = {file.split('_')[-2]: pd.read_csv(os.path.join(path, file)) for file in os.listdir(path)}
    svg = from_contents({k: set(v['genes'].values) for (k,v) in data.items()})
    ax_dict = UpSet(svg, subset_size='count', show_counts=True)
    fig = plt.figure()
    ax_dict.plot(fig)
    plt.savefig(os.path.join(out_path, 'upsetplot.png'))
    