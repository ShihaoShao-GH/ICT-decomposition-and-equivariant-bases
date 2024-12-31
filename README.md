# ICT-decomposition-and-equivariant-bases

This is the official code repository of the paper *High-Rank Irreducible Cartesian Tensor Decomposition and
Bases of Equivariant Spaces* (arXiv: https://arxiv.org/abs/2412.18263). It can generate high-order ICT decomposition matrices. In our platform, the $n=6,\dots,9$ ICT decomposition matrices are obtained in 1s, 3s, 11s, and 4m32s via our algorithm, respectively. This algorithm is based on obtaining analytical expressions but not a numerical solver, so it is very efficient. The bases of $Hom_{O(3)}(S,S^\prime)$ and $Hom_{SU(2)}(S,S^\prime)$ can also be efficiently obtained by our algorithm. If you want to work with $SU(n)$, then you need to visit https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/ website to obtain the required CG coefficients and do some engineering.

The only external dependency packages are PyTorch and e3nn. The e3nn package can be easily installed using the command `pip install --upgrade e3nn`. The code can run on any computer with Python version $>$ 3.0.0 and PyTorch version $>$ 1.0. The code for generating the basis for $SU(2)$ does not require the e3nn package. However, the code for generating Clebsch-Gordan (CG) coefficients is borrowed from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html. The CG coefficients for $SU(n)$ can be obtained from https://homepages.physik.uni-muenchen.de/~vondelft/Papers/ClebschGordan/. The code for generating these coefficients is detailed on pages 19–34 of their paper (arXiv: https://arxiv.org/pdf/1009.0437). 

## Usage

`ICT_decomp.py`: ICT decomposition matrices generation (Algorithm 1 in the paper).

`equivariant_bases.py`: Generate equivariant basis for $Hom_{O(3)}(S,S^\prime)$ (Algorithm 2).

`equivariant_bases_end.py`: Generate equivariant basis for $End_{O(3)}((ℝ^3)^{\otimes n})$.

`equivariant_bases_su(2).py`: Generate equivariant basis for $Hom_{SU(2)}(S,S^\prime)$ (Algorithm 2).

`*_algo4.py`: Efficient implementation with Algorithm 4.

## Contact

If you have any techniqual question, please open an issue or send email to shaoshihao@pku.edu.cn.
