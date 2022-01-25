import os
import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt

import transfer_matrix

"""
This script should be used to produce all the plots needed for the paper.  
"""
# find the folder of the module
folder = path.dirname(path.abspath(transfer_matrix.__file__))
sys.path.append(path.dirname(folder))
os.chdir(folder)

## start by loading a matrix:
T = transfer_matrix.TransferMatrix.from_file(folder+'/matrix.npz')

## plot some example images

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num=10)

# TODO: need to get unified single map for both images together, and pass the legend argument into single_geometry.

plt.sca(ax1)
transfer_matrix.single_geometry(source_radius=0.8, distance=0, occulter_radius=0, plotting=True, legend=False)

plt.sca(ax2)
transfer_matrix.single_geometry(source_radius=0.8, distance=0.4, occulter_radius=0, plotting=True, legend=False)

plt.sca(ax3)
transfer_matrix.single_geometry(source_radius=0.8, distance=1.6, occulter_radius=0, plotting=True, legend=True)
