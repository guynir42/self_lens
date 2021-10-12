import pytest

import numpy as np
import matplotlib.pyplot as plt

import transfer_matrix


@pytest.fixture
def matrix():
    T = transfer_matrix.TransferMatrix()
    T.load('matrix.npz')
    return T


