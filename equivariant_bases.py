import torch
from e3nn import o3
from tqdm import tqdm
from typing import Tuple, List
# space_1 = {"space":[[1,2],[1,3]],"parity":-1} # [1,2],[1,3] means (1 \otimes 2) \oplus (1 \otimes 3)
# space_2 = {"space":[[1,3]],"parity":-1}
def general_equivariant_basis_generator(space_1: Tuple, space_2: Tuple) \
    -> Tuple[List[Tuple] ,List[torch.Tensor]]:
    path_list_composition = [] 
    path_list_decomposition = []
    def general_scheme(rank_list, this_path, current_rank, ind, path_list, parity, index):
        rank = rank_list[ind]
        for i in range(abs(current_rank - rank), current_rank + rank + 1):
            this_path_ = this_path.copy()
            this_path_.append([rank,i])
            if ind+1 >= len(rank_list):
                path_list.append({"path": this_path_, "parity": parity, "index": index})
            else:
                general_scheme(rank_list, this_path_, i, ind+1, path_list, parity, index)
    for n,i in enumerate(space_1["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_decomposition, space_1["parity"], n)
    for n,i in enumerate(space_2["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_composition, space_1["parity"], n)
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
