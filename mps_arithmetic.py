import numpy as np


class MPSaddition():
    """Class for adding two MPS with open boundary condition."""

    def dsum(self, A, B):
        d = A.shape[1]
        assert d == B.shape[1]

        dsum = np.zeros((A.shape[0] + B.shape[0], d, A.shape[2] + B.shape[2]))
        dsum[:A.shape[0], :, :A.shape[2]] = A
        dsum[A.shape[0]:, :, A.shape[2]:] = B

        return dsum


    def row(self, A, B):
        assert A.shape[0] == B.shape[0] == 1
        d = A.shape[1]
        assert d == B.shape[1]

        row = np.zeros((1, d, A.shape[2] + B.shape[2]))
        row[:, :, :A.shape[2]] = A
        row[:, :, A.shape[2]:] = B

        return row


    def col(self, A, B):
        assert A.shape[2] == B.shape[2] == 1
        d = A.shape[1]
        assert d == B.shape[1]

        col = np.zeros((A.shape[0] + B.shape[0], d, 1))
        col[:A.shape[0], :, :] = A
        col[A.shape[0]:, :, :] = B

        return col


    def add(self, mps_a, mps_b):
        L = len(mps_a)
        assert len(mps_b) == L

        mps = [self.row(mps_a[0], mps_b[0])] + [self.dsum(mps_a[i], mps_b[i]) for i in range(1, L - 1)] + [self.col(mps_a[L - 1], mps_b[L - 1])]
        
        return mps