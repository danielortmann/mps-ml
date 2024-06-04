# Machine Learning with Matrix Product States
This repository implements an algorithm based on matrix product states (MPSs) for supervised learning. Conventional machine learning techniques usually involve the iterative minimization of a problem-specific loss function using gradient descent methods. As an alternative approach, this algorithm utilizes tensor network operations to prepare a collection of matrix product states that serve as a machine learning model for a classification task. The algorithm iteratively compresses and adds MPSs that encode the input data. The algorithm is benchmarked on the MNIST dataset for varying virtual bond dimensions and shows competitive results.

The algorithm has a computational cost $\mathcal{O}(N L \chi^{2})$, where $N$, $L$, and $\chi$ denote the number of images in the training set, the number of pixels per image and the virtual bond dimension of the MPSs, respectively.

## Data Encoding and ML Model Construction

The MNIST dataset contains 60,000 images of handwritten digits between 0 and 9 in $28\hspace{0.01em}\times\hspace{0.01em}28$ pixel grayscale format and has been used for model benchmarking in a variety of machine learning research tasks. One pixel takes values ranging from 0 to 255, where 0 corresponds to a black pixel and 255 to a white pixel.

We perform two pre-processing steps. First, we apply a mean filter to each image in the dataset, i.e. we course-graining each image down to a size of $14\hspace{0.01em}\times\hspace{0.01em}14$ pixel by averaging over each $2\hspace{0.01em}\times\hspace{0.01em}2$ pixel. Second, we rescale each pixel value to $[0, 1]$. Each image of a digit $y\in\\{0, ..., 9\\}$ then corresponds to a vector $x\in[0, 1]^{L}$ with $L = 14\cdot14 = 196$ pixels $x_{j}\in[0, 1]$. The resulting dataset is denoted by $\mathcal{D} = \\{(x^{(n)},\, y^{(n)})\\} \subset [0, 1]^{L} \times \\{0, ..., 9\\}$, containing $N_{\text{data}}$ samples.
Then each image is mapped to a product state of $L$ spins by applying the feature map

  $$ \Phi: [0, 1]^{L} \to \left({\mathbb{C}^{2}}\right)^{\otimes{L}} \hspace{0.05cm}, \quad x \mapsto\ket{\Phi(x)} = \bigotimes_{j=1}^{L}  \hspace{0.1cm} \ket{\phi^{j}(x_{j})} \hspace{0.1cm}, $$

with the local feature map

  $$ \ket{\phi^{j}(x_{j})} = \cos(\frac{\pi}{2}x_{j}) \hspace{0.1cm} \ket{0} + \sin(\frac{\pi}{2}x_{j}) \hspace{0.1cm} \ket{1} \hspace{0.1cm}. $$

This means each image is encoded as a vector in a vector space whose dimension is exponentially large in the number of pixels in the image. Then, for each digit $d\in\\{0, ..., 9\\}$, we construct the superpostion of all samples

  $$ \ket{\Sigma_{d}} = \sum_{\\{n \hspace{0.05cm}|\hspace{0.05cm} y^{(n)}=d\\}} \ket{\Phi(x^{(n)})} \hspace{0.1cm}, $$

which is normalized to one,

  $$ \ket{\hat{\Sigma}\_{d}} = \frac{\ket{\Sigma_{d}}}{\sqrt{\bra{\Sigma_{d}}\ket{\Sigma_{d}}}} \hspace{0.1cm}. $$

The machine learning model is then defined by the collection $\\{\ket{\hat{\Sigma}_{d}} \hspace{0.05cm}|\hspace{0.1cm} d\in\\{0, ..., 9\\}\\}$, where an image $x\in\\{0, 1\\}^{L}$ is classified by
  
  $$ \mathrm{arg}\hspace{0.1cm}\mathrm{max} \_{d\in\\{0, ..., 9\\}} \hspace{0.2cm} |\langle{\Phi(x)}|\hat{\Sigma}_{d}\rangle| \hspace{0.1cm}. $$

In practice, the state $\ket{\Sigma_{d}}$ could be constructed by summing up all product states that encode an image of digit $d$ in the dataset. This can be achieved by basic MPS arithmetic [U. Schollwoeck, 2011]. The addition of two MPSs $\ket{\psi}$ and $\ket{\phi}$ of length $L$ and with open boundary conditions,

  $$ \ket{\psi} = \sum_{\vec{\sigma}} M^{\sigma_{1}}...M^{\sigma_{L}}\ket{\vec{\sigma}} $$
  
  $$ \ket{\psi} = \sum_{\vec{\sigma}} \tilde{M}^{\sigma_{1}}...\tilde{M}^{\sigma_{L}}\ket{\vec{\sigma}} \hspace{0.1cm}, $$

is given by

  $$ \ket{\psi} + \ket{\phi} = \sum_{\vec{\sigma}} N^{\sigma_{1}}...N^{\sigma_{L}}\ket{\vec{\sigma}} \hspace{0.1cm},$$

where

$$ N^{\sigma_{i}} =
\begin{cases}
    M^{\sigma_{i}} \oplus \tilde{M}^{\sigma_{i}} & \quad \text{for} \hspace{0.25em} i \in \\{2, ...\,, L-1 \\} \\
    [M^{\sigma_{i}},\, \tilde{M}^{\sigma_{i}}] & \quad \text{for} \hspace{0.25em} i=1 \\
    [M^{\sigma_{i}},\, \tilde{M}^{\sigma_{i}}]^{\mathrm{T}} & \quad \text{for} \hspace{0.25em} i=L
\end{cases} \hspace{0.1cm}. $$

Note that $N^{\sigma_{1}}$ and $N^{\sigma_{L}}$ are row and column vectors, respectively. Since each image is encoded in a product state, its MPS has virtual bond dimension one, i.e. $M^{\sigma_{i}}$ contains a single value. Therefore, with each addition, the virtual bond dimension increases by one and the summation over all samples yields $N^{\sigma_{i}}$ (for $i\in\\{2, ..., L-1\\}$) as a diagonal matrix, where each entry corresponds to exactly one sample. Naively, this would require to store $\ket{\Sigma_{d}}$ as an MPS with virtual bond dimension equal to the number of samples in the dataset with digit $d$, which is about 6,000 in MNIST. This is of course completely impractical. Hence, we have to compress our MPS during summation to a reasonable virtual bond dimension $\chi$. This can be achieved as follows:

## Compression Algorithm

We divide our dataset into $N_{\text{data}}/\chi$ batches. Adding two batches together yields an MPS with virtual bond dimension of $2\cdot\chi$, which we compress back to a virtual bond dimension of $\chi$. Then we add the next batch to the resulting MPS and compress again. This is done iteratively until all batches have been added.

The compression is performed as follows: At site $k$ we perform a singular value decomposition, discard the $\chi$ smallest singular values and re-absorb the singular value diagonal and the isometry containing the right-singular vectors into the tensor at site $k+1$. The isometry containing the left-singular vectors defines the updated tensor at site $k$. This procedure is done at each site $k = 1, ..., L$ from left to right, bringing the MPS into right-canonical form, i.e. the state is normalized. Therefore, to prepare the state in Eq.~\ref{eq:sigma}, the MPS must be re-normalized after each compression to its original norm $\mathcal{N}$, such that each image enters the summation with the same weight. By defining a dummy tensor of shape $(1\hspace{0.01em}\times\hspace{0.01em}1\hspace{0.01em}\times\hspace{0.01em}1)$ at site $L+1$, the compression algorithm stores the norm of the MPS (before it has been compressed) in this dummy tensor. Multiplying each tensor (after compression) by $\mathcal{N}^{1/L}$, then recovers the unnormalized (but compressed) state. This re-normalization is applied at each summation step, except for the last one to ensure the normalization of the final state $\ket{\hat{\Sigma}\_{d}}$. By this procedure, we have constructed an approximation

  $$ \ket{\hat{\Sigma}\_{d}^{\chi}} \approx \ket{\hat{\Sigma}\_{d}} \hspace{0.1cm}. $$

We note that, by performing compression, we lose the diagonal structure in the virtual space and obtain dense MPS tensors instead.

## Reference

J. Martyn, G. Vidal, C. Roberts, S. Leichenauer, "Entanglement and tensor networks for supervised image classification," arXiv e-prints, arXiv:2007.06082 (2020), [arXiv:2007.06082 [quant-ph]](https://arxiv.org/abs/2007.06082).
