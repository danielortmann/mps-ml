# Machine Learning with Matrix Product States
This repository implements an algorithm based on matrix product states (MPSs) for supervised learning. Conventional machine learning techniques usually involve the iterative minimization of a problem-specific loss function using gradient descent methods. As an alternative approach, this algorithm utilizes tensor network operations to prepare a collection of matrix product states that serve as a machine learning model for a classification task. The algorithm iteratively compresses and adds MPSs that encode the input data. The algorithm is benchmarked on the MNIST dataset for varying virtual bond dimensions and shows competitive results.

The algorithm has a computational cost $\mathcal{O}(N L \chi^{2})$, where $N$, $L$, and $\chi$ denote the number of images in the training set, the number of pixels per image and the virtual bond dimension of the MPSs, respectively.

## Reference

J. Martyn, G. Vidal, C. Roberts, S. Leichenauer, "Entanglement and tensor networks for supervised image classification," arXiv e-prints, arXiv:2007.06082 (2020), [arXiv:2007.06082 [quant-ph]](https://arxiv.org/abs/2007.06082).