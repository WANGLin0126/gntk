import math
import numpy as np
import torch
import scipy as sp

class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(scale in ['uniform', 'degree'])
    
    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = torch.sqrt(torch.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = torch.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - torch.arccos(S)) / math.pi
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / torch.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        return torch.sparse.mm(adj_block, S.reshape(-1,1)).reshape(N, N) * scale_mat

    def sparse_kron(self, A, B):
        """
        A, B: torch.sparse.FloatTensor of shape (m, n) and (p, q)
        Returns: the Kronecker product of A and B
        """
        A = A.coalesce()
        B = B.coalesce()
        m, n = A.shape
        p, q = B.shape
        n_A  = A._nnz()
        n_B  = B._nnz()

        indices_A = A.indices()
        indices_B = B.indices()
        indices_A[0,:] = indices_A[0,:] * p
        indices_A[1,:] = indices_A[1,:] * q

        indices = (indices_A.repeat(n_B, 1) + indices_B.t().reshape(2*n_B,1))
        ind_row = indices.index_select(0,torch.arange(start = 0, end = 2*n_B, step = 2, device=A.device) ).reshape(-1)
        ind_col = indices.index_select(0,torch.arange(start = 1, end = 2*n_B, step = 2, device=A.device) ).reshape(-1)

        new_ind = torch.cat((ind_row, ind_col)).reshape(2, n_A*n_B)
        values = torch.ones(n_A*n_B).to(A.device)
        new_shape = (m * p, n * q)
        
        return torch.sparse_coo_tensor(new_ind, values, new_shape)

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = torch.clip(S, -1, 1)
        DS = (torch.pi - torch.arccos(S)) / math.pi
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / torch.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        return torch.sparse.mm(adj_block,S.reshape(-1,1)).reshape(N1, N2) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g
        A: adjacency matrix
        """
        N = A.shape[0]
        aggr_optor = self.sparse_kron(A, A)
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = (1./torch.sparse.sum(aggr_optor,1).to_dense()).reshape(N,N)

        diag_list = []
        adj_block = self.sparse_kron(A, A)

        # input covariance
        sigma = torch.matmul(g.node_features, g.node_features.T)
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
        ntk = sigma.clone()
		
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)
        return diag_list

    def gntk(self, g1, g2, diag_list1, diag_list2, A1, A2):
        """
        compute the GNTK value \Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """
        
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        
        adj_block = self.sparse_kron(A1, A2)

        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = (1./torch.sparse.sum(adj_block,1).to_dense()).reshape(n1,n2)
        
        jump_ntk = 0
        sigma = torch.matmul(g1.node_features, g2.node_features.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
        ntk = sigma.clone()
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)
        if self.jk:
            return torch.sum(jump_ntk) * 2
        else:
            return torch.sum(ntk) * 2
