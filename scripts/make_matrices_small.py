# use this script to make saved/matrix_SR0.010-0.100_D0.000-2.200.npz for very small sources.
# running the script as is could take several days due to the high resolution (=1000)
# thus the recommended use is to split this up to multiple workers, using the bash script matrix_factory.sh:
# $ source scripts/matrix_factory.sh
# That should split up the matrix into matrices for different distances (in 0.2 intervals).
# To combine the 10 output matrices use the combine_matrices.py script:
# $ python scripts/combine_matrices.py saved/fragments/matrix_SR0.010-0.100_D*.npz -o saved/matrix_SR0.010-0.100_D0.000-2.000.npz
# NOTE: all paths are relative to the base dir (if you are inside the scripts folder, they need to be updated).
# When run in parallel on 11 workers, the job could be done in a few hours.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transfer_matrix import TransferMatrix

print(sys.argv)

T = TransferMatrix()
T.min_source = 0.01
T.max_source = 0.1
T.step_source = 0.005
T.max_dist = 3.0
T.step_dist = 0.005
T.max_occulter = 2.0
T.step_occulter = 0.005
T.num_pixels = 1e6
T.num_points = 1e6

if len(sys.argv) > 2:
    T.min_dist = float(sys.argv[1])
    T.max_dist = float(sys.argv[2])
if len(sys.argv) > 3:
    T.step_dist = float(sys.argv[3])

if not os.path.exists('saved/fragments'):
    os.makedirs('saved/fragments')

filename = f'saved/fragments/matrix_SR{T.min_source:.3f}-{T.max_source:.3f}_D{T.min_dist:.3f}-{T.max_dist:.3f}'

T.make_matrix()
T.save(filename)

