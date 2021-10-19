import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transfer_matrix import TransferMatrix

print(sys.argv)

T = TransferMatrix()
T.min_source = 0.1
T.max_source = 1
T.step_source = 0.05
T.max_dist = 5
T.step_dist = 0.1
T.max_occulter = 1.5
T.step_occulter = 0.05

if len(sys.argv) > 2:
    T.min_dist = float(sys.argv[1])
    T.max_dist = float(sys.argv[2])
if len(sys.argv) > 3:
    T.step_dist = float(sys.argv[3])

filename = f'matrix_SR{T.max_source:.2f}_D{T.min_dist}_D{T.max_dist}'

T.make_matrix()
T.save(filename)

