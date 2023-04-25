import sys
import logging

from .clustering_algorithm import ClusteringAlgorithm
from .scc import SccAlgo
from .utils import timeit

try:
    from .leiden_louvain import LeidenLouvainAlgo
except ImportError:
    logging.error("Cannot import LeidenLouvainAlgo.")

try:
    from .spagft import SpagftAlgo
except ImportError:
    logging.warn("Module SpaGFT is not installed. Run: pip install SpaGFT==0.1.1b0   - if you wish to use it.")

try:
    from .spatialde import SpatialdeAlgo
except ImportError:
    logging.warn("Module SpatialDE is not installed. Run: pip install spatialde   - if you wish to use it.")

try:
    from .hotspot import HotspotAlgo
except ImportError:
    logging.warn("Module hotspot is not installed. It is used by stereopy so you can un: pip install stereopy   - if you wish to use it.")

try:
    from .spagcn import SpagcnAlgo
except ImportError:
    logging.warn("Module SpaGCN is not installed. Run: pip install SpaGCN   - if you wish to use it.")


