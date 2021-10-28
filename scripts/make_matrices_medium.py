
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transfer_matrix import TransferMatrix

print(sys.argv)

T = TransferMatrix()
T.min_source = 0.1
T.max_source = 1.0
T.step_source = 0.025
T.max_dist = 10.0
T.step_dist = 0.025
T.max_occulter = 5.0
T.step_occulter = 0.025
T.pixels = 1e7
T.num_points = 1e5

if len(sys.argv) > 2:
    T.min_dist = float(sys.argv[1])
    T.max_dist = float(sys.argv[2])
if len(sys.argv) > 3:
    T.step_dist = float(sys.argv[3])

filename = f'saved/matrix_SR{T.min_source:.3f}-{T.max_source:.3f}_D{T.min_dist:.3f}-{T.max_dist:.3f}'

T.make_matrix()
T.save(filename)

