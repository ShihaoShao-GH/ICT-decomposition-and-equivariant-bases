import torch
from tqdm import tqdm
from typing import Tuple, List, Union
# Taken from e3nn
# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
def su2_clebsch_gordan(j1: Union[int, float], j2: Union[int, float], j3: Union[int, float]) -> torch.Tensor:
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = torch.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype=torch.float64)
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = _su2_clebsch_gordan_coeff(
                        (j1, m1), (j2, m2), (j3, m1 + m2)
                    )
    return mat
def _su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    from fractions import Fraction
    from math import factorial
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))
    def f(n: int) -> int:
        assert n == round(n)
        return factorial(round(n))
    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v), f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3)
        )
    C = C * S
    return C
# space_1 = {"space":[[1,2],[1,3]],"parity":-1} # [1,2],[1,3] means (0.5 \otimes 1) \oplus (0.5 \otimes 1.5)
# space_2 = {"space":[[1,3]],"parity":-1}
def general_equivariant_basis_generator_SU2(space_1: Tuple, space_2: Tuple) \
    -> Tuple[List[Tuple] ,List[torch.Tensor]]:
    path_list_composition = [] 
    path_list_decomposition = []
    def general_scheme(rank_list, this_path, current_rank, ind, path_list, parity, index):
        rank = rank_list[ind]
        for i in range(abs(current_rank - rank), current_rank + rank + 1, 2):
            this_path_ = this_path.copy()
            this_path_.append([rank,i])
            if ind + 1 >= len(rank_list):
                path_list.append({"path": this_path_, "parity": parity, "index": index})
            else:
                general_scheme(rank_list, this_path_, i, ind + 1, path_list, parity, index)
    for n,i in enumerate(space_1["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_decomposition, space_1["parity"], n)
    for n,i in enumerate(space_2["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_composition, space_2["parity"], n)
    def path_matrices_generators(path):
        path_matrix = su2_clebsch_gordan(0,0,0)
        current_j = 0
        for (bridge_j, next_j) in path:
            cg_matrix = su2_clebsch_gordan(bridge_j/2.,current_j/2.,next_j/2.)
            path_matrix = torch.einsum("abc,dce->dabe", path_matrix, cg_matrix)\
                .reshape(bridge_j+1, -1, next_j+1)
            current_j = next_j
        path_matrix = path_matrix.reshape(-1,path_matrix.shape[-1])
        path_matrix = path_matrix*(1./(path_matrix**2).sum(0)[0]**(0.5)) # normalize
        return path_matrix
    equivariant_basis = []
    paths_and_spaces = []
    for path_c in path_list_composition:
        for path_d in path_list_decomposition:
            if path_d["path"][-1][-1] == path_c["path"][-1][-1] and \
                path_d["parity"] == path_c["parity"]:
                    paths_and_spaces.append({"path_c": path_c, "path_d": path_d})
                    path_matrix_c = path_matrices_generators(path_c["path"])
                    path_matrix_d = path_matrices_generators(path_d["path"])
                    equivariant_basis.append(path_matrix_c @ path_matrix_d.T)
    return paths_and_spaces, equivariant_basis
