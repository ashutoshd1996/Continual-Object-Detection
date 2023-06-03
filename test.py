
import torch

def load_fisher(path):
    print("\nLoading fisher matrix from : ", path)
    
    fisher_matrix = torch.load(path)
    print(fisher_matrix)
    # return fisher_matrix

load_fisher('./logs/cl_logs/Fisher_Matrix/M1_fisher.pt')

