import torch
from e3nn import o3
from tqdm import tqdm
from typing import Tuple, List
def equivariant_basis_generation(n_total : int) -> List[torch.Tensor]:
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = o3.wigner_3j(0,0,0)
    pathmatrices_list = []
    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now<=n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now==0 and (j!=1) ) and n_now+1<=n_total:
                    cgmatrix = o3.wigner_3j(1,j_now,j)
                    this_pathmatrix_ = torch.einsum("abc,dce->dabe",this_pathmatrix,cgmatrix)
                    this_pathmatrix_ = this_pathmatrix_.reshape(cgmatrix.shape[0],-1,cgmatrix.shape[-1])
                    paths_generate(n_now+1,j,this_path.copy(),this_pathmatrix_,n_total)
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1,this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix*(1./(this_pathmatrix**2).sum(0)[0]**(0.5)) # normalize
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return     
    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    equiv_basis_matrix = torch.zeros([3**n_total,3**n_total])
    param_list = []
    for ind, path in tqdm(enumerate(path_list),total=len(path_list)):
        for ind2, path2 in enumerate(path_list):
            if path[-1] == path2[-1] and ind !=ind2:
                parameter = torch.nn.Parameter(torch.rand(1))
                equiv_basis_matrix+=(pathmatrices_list[ind] @ pathmatrices_list[ind].T)*parameter
                param_list.append(parameter)
    return equiv_basis_matrix, param_list
