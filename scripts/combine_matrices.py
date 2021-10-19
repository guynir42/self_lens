import sys, getopt
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transfer_matrix import TransferMatrix

output_filename = 'matrix.npz'

opts, args = getopt.getopt(sys.argv, 'hi:o:')

for opt, arg in opts:
    if opt == '-h':
        print('combine_matrices.py <files_to_combine> -o <output file>')
    if opt == '-o':
        output_filename = arg

for filename in args[1:]:
    print(f'input file: {filename}')

print(f'output file: {output_filename}')

T = TransferMatrix()

