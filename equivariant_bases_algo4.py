import torch
from e3nn import o3
from tqdm import tqdm
from typing import Tuple, List
from collections import Counter
# space_1 = {"space":[[1,2],[1,3]],"parity":-1} # [1,2],[1,3] means (1 \otimes 2) \oplus (1 \otimes 3)
# space_2 = {"space":[[1,3]],"parity":-1}
def general_equivariant_basis_generator(space_1: Tuple, space_2: Tuple) \
    -> Tuple[List[Tuple] ,List[torch.Tensor]]:
    path_list_composition = [] 
    path_list_decomposition = []
    last_path_parity_comp = []
    last_path_parity_decomp = []
    def general_scheme(rank_list, this_path, current_rank, ind, path_list, parity, index, last_path_parity):
        rank = rank_list[ind]
        for i in range(abs(current_rank - rank), current_rank + rank + 1):
            ind_ = ind
            this_path_ = this_path.copy()
            this_path_.append([rank,i])
            ind_ +=1
            if ind_ >= len(rank_list):
                path_list.append({"path": this_path_, "parity": parity, "index": index})
                last_path_parity.append((this_path_[-1][-1],parity))
            else:
                general_scheme(rank_list, this_path_, i, ind_, path_list, parity, index, last_path_parity)
    for n,i in enumerate(space_1["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_decomposition, space_1["parity"], n, last_path_parity_decomp)
    for n,i in enumerate(space_2["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_composition, space_2["parity"], n, last_path_parity_comp)
    def path_matrices_generators(path):
        path_matrix = o3.wigner_3j(0,0,0)
        current_j = 0
        for (bridge_j, next_j) in path:
            cg_matrix = o3.wigner_3j(bridge_j,current_j,next_j)
            path_matrix = torch.einsum("abc,dce->dabe", path_matrix, cg_matrix)\
                .reshape(2*bridge_j+1, -1, 2*next_j+1)
            current_j = next_j
        path_matrix = path_matrix.reshape(-1,path_matrix.shape[-1])
        path_matrix = path_matrix*(1./(path_matrix**2).sum(0)[0]**(0.5)) # normalize
        return path_matrix
    CT_matrix_list = []
    C_matrix_list = []
    fc_list = []
    for path_d in path_list_decomposition:
        CT_matrix_list.append(path_matrices_generators(path_d["path"]))
    for path_c in path_list_composition:
        C_matrix_list.append(path_matrices_generators(path_c["path"]))
    for k_d,v_d in Counter(last_path_parity_decomp).items():
        for k_c,v_c in Counter(last_path_parity_comp).items():
            if k_d == k_c:
                fc_list.append(torch.nn.Linear(v_c,v_d,bias=False))
    return fc_list, CT_matrix_list, C_matrix_list
