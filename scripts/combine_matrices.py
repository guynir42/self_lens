import sys
import getopt
from os import path
import re

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transfer_matrix import TransferMatrix

output_filename = "matrix.npz"

opts, args = getopt.gnu_getopt(sys.argv[1:], "ho:")

for opt, arg in opts:
    if opt == "-h":
        print("combine_matrices.py <files_to_combine> -o <output file>")
    if opt == "-o":
        output_filename = arg

# sort args by distance
def dist(filename):
    a = re.search(r"D\d+\.\d+", filename)
    if a is not None:
        return float(a.group(0)[1:])
    else:
        return 0


args = sorted(args, key=lambda f: dist(f))

for filename in args:
    print(f"input file: {filename}")

print(f"output file: {output_filename}")

T0 = TransferMatrix()

for f in args:
    T = TransferMatrix()
    T.load(f)
    T0 = T0 + T

T0.save(output_filename)
