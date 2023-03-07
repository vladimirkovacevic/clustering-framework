import argparse as ap
import logging
import os

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='A script that performs clustering with tissue modules identified using SpaGFT')
    parser.add_argument('-f', '--file', help='File that contain data to be clustered', type=str, required=True)
    parser.add_argument('-o', '--out_path', help='Absolute path to store outputs', type=str, required=True)
    parser.add_argument('-v', '--verbose', help='Show logging messages', action='count', default=0)

    args = parser.parse_args()

    if args.verbose == 0:
        logging.basicConfig(level=logging.WARNING, force=True)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)