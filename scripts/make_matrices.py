import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transfer_matrix import TransferMatrix

print(sys.argv)

T = TransferMatrix()
T.min_source = 0.01
T.max_source = 0.01
T.step_source = 0.005
T.min_dist = 4.9
T.max_dist = 5
T.step_dist = 0.005
T.max_occulter = 3
T.step_occulter = 0.005
T.num_points = 1e5
T.num_pixels = 3e7

if len(sys.argv) > 2:
    T.min_dist = float(sys.argv[1])
    T.max_dist = float(sys.argv[2])
if len(sys.argv) > 3:
    T.step_dist = float(sys.argv[3])

filename = f'saved/matrix_SR{T.min_source:.3f}-{T.max_source:.3f}_D{T.min_dist:.3f}-{T.max_dist:.3f}'

T.make_matrix()
# T.save(filename)

