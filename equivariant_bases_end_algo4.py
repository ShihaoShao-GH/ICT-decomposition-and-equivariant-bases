import torch
from e3nn import o3
from tqdm import tqdm
from collections import Counter

def change_of_basis_generation(n_total):
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = o3.wigner_3j(0,0,0)
    pathmatrices_list = []
    # generate paths
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now<=n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not j > n_now+1 and not (j_now==0 and (j!=1) ) and n_now+1<=n_total:
                    wigner3j = o3.wigner_3j(1,j_now,j)
                    this_pathmatrix_ = torch.einsum("abc,dce->dabe",this_pathmatrix,wigner3j)
                    this_pathmatrix_ = this_pathmatrix_.reshape(wigner3j.shape[0],-1,wigner3j.shape[-1])
                    paths_generate(n_now+1,j,this_path.copy(),this_pathmatrix_,n_total)
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1,this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix*(1./(this_pathmatrix**2).sum(0)[0]**(0.5)) # normalize
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return     
    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    # sort 
    def get_last_digit(s):
        return int(s[-1])  
    sorted_indices = sorted(range(len(path_list)), key=lambda i: get_last_digit(path_list[i]))
    sorted_path_list = [path_list[i][-1]*2+1 for i in sorted_indices]
    sorted_pathmatrices_list = [pathmatrices_list[i] for i in sorted_indices]
    CT_matrix = torch.concat(sorted_pathmatrices_list,-1)
    fc_list = torch.nn.ModuleList()
    for _,v in Counter(sorted_path_list).items():
        fc_list.append(torch.nn.Linear(v,v,bias=False))
    return sorted_path_list, fc_list, CT_matrix
