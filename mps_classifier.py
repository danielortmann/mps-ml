import numpy as np
import scipy as sp

from mps_arithmetic import MPSaddition



class MPSclassifier(MPSaddition):
    """Class for building MPS classifier."""

    def __init__(self, chi):
        self.chi = chi
        self.trained = False
        self.model = None


    def mean_filter(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(14, 2, 14, 2).transpose(0, 2, 1, 3).mean(axis=(2, 3))


    def image_to_mps(self, x: np.ndarray) -> list:
        mps = list()
        for pixel in x.flatten():

            M = np.zeros([1, 2, 1], float)
            M[0, 0, 0] = np.cos(np.pi/2*pixel/255)
            M[0, 1, 0] = np.sin(np.pi/2*pixel/255)
            mps.append(M)

        return mps


    def compress(self, mps: list, chi: int, normalize: bool = False) -> tuple[list, float]:
        L = len(mps)
        norm = np.array([[[1]]])
        mps = mps + [norm]
        for n in range(L):

            # SVD and truncation
            vL, d, vR = mps[n].shape
            M = mps[n].reshape(vL*d, vR)
            U, S, Vh = sp.linalg.svd(M, full_matrices=False, lapack_driver='gesvd')
            U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
            rank = len(S)
            mps[n] = U.reshape(vL, d, rank)

            # absorb SVD tensors into next tensor
            vL, d, vR = mps[n+1].shape
            M = mps[n+1].reshape(vL, d*vR)
            mps[n+1] = ((S[:, None] * Vh) @ M).reshape(rank, d, vR)

        # flip signs of last two tensors if norm negative
        if mps[-1].item().real < 0:
            mps[-1] *= -1
            mps[-2] *= -1
        norm = mps[-1].item().real
        del mps[-1]

        # re-normalize to original norm
        if not normalize:
            mps = [M * norm**(1/L) for M in mps]

        return mps, norm


    def train(self, X: np.ndarray, Y: np.ndarray):
        model = list()
        for digit in range(10):

            samples = X[Y == digit]
            for n, x in enumerate(samples):

                x = self.image_to_mps(self.mean_filter(x))
                state = self.add(state, x) if n > 0 else x

                if n + 1 == len(samples):
                    state, _ = self.compress(state, self.chi, normalize=True)

                elif (n + 1)%self.chi == 0 and n + 1 >= 2*self.chi:
                    state, _ = self.compress(state, self.chi, normalize=False)

            model.append(state)

        self.model = model


    def inner_product(self, mps_a: list, mps_b: list) -> float:
        L = len(mps_a)
        assert L == len(mps_b)

        t = np.tensordot(mps_b[0], mps_a[0].conj(), axes=[1, 1])
        t = t.squeeze(axis=(0, 2))
        for n in range(1, L):

            t = np.tensordot(t, mps_b[n], axes=[0, 0])
            t = np.tensordot(t, mps_a[n].conj(), axes=[[0, 1], [0, 1]])

        return t.item()


    def project(self, x: np.ndarray) -> list:
        assert self.model is not None
        projections = [abs(self.inner_product(state, self.image_to_mps(self.mean_filter(x)))) for state in self.model]

        return projections


    def predict(self, x: np.ndarray) -> int:
        projections = self.project(self.model, x)
        prediction = np.argmax(projections)

        return prediction


    def compute_accuracies(self, X: np.ndarray, Y: np.ndarray) -> list:
        accuracies = list()
        for digit in range(10):

            samples = X[Y==digit]
            projections = [self.project(x) for x in samples]
            predictions = np.argmax(projections, axis=1)
            accuracy = sum(predictions == digit) / len(samples)
            accuracies.append(accuracy)

        return accuracies