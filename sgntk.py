import torch
import torch.nn as nn

class SimplifyingGraphNeuralTangentKernel(nn.Module):
    def __init__(self,K=2, L=2 ):
        super(SimplifyingGraphNeuralTangentKernel, self).__init__()
        self.K = K
        self.L = L

    def updat_sigma(self, Sigma_xx, Sigma_XX, Sigma_xX):
        """
        update the sigma matrix
        """
        a_xX  = torch.clip(2 * Sigma_xX/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)
        a_xx  = torch.clip(2 * Sigma_xx/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_xx)),-1,1)
        a_XX  = torch.clip(2 * Sigma_XX/(torch.sqrt(1 + 2*Sigma_XX)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)

        Sigma_xX = 2/torch.pi * torch.asin(a_xX)
        Sigma_xx = 2/torch.pi * torch.asin(a_xx)
        Sigma_XX = 2/torch.pi * torch.asin(a_XX)

        if Sigma_xX.isnan().any():
            raise ValueError('NaN in Sigma_xX')
        elif Sigma_xx.isnan().any():
            raise ValueError('NaN in Sigma_xx')
        elif Sigma_XX.isnan().any():
            raise ValueError('NaN in Sigma_XX')
        

        return  Sigma_xx, Sigma_XX, Sigma_xX
    
    def GCF(self, x, A, k=1):
        """
        Graph Convolutional Filtering
        """
        A = torch.clip(A + torch.eye(A.shape[0]).to(A.device),0,1)
        A = A + torch.eye(A.shape[0]).to(A.device)
        D = torch.diag(torch.sum(A, 1))
        D = torch.inverse(torch.sqrt(D))
        A = torch.matmul(torch.matmul(D, A), D)
        for i in range(k):
            A = torch.matrix_power(A, k)
        x = torch.matmul(A, x)
        return x

    def similarity(self, g1, g2, A1, A2):
        x, X = g1.node_features, g2.node_features
        A1 ,A2 = A1.to_dense(), A2.to_dense()
        x, X = self.GCF(x, A1, self.K), self.GCF(X, A2, self.K)

        Sigma_xx    = torch.diag(torch.matmul(x, x.t())).reshape(-1,1)
        Sigma_XX    = torch.diag(torch.matmul(X, X.t())).reshape(1,-1)
        Sigma_xX  = torch.matmul(x, X.t())
        
        Sigma_xX_list = []
        Sigma_xX_list.append(Sigma_xX)
        for l in range(self.L):
            Sigma_xx, Sigma_XX, Sigma_xX =  self.updat_sigma(Sigma_xx, Sigma_XX, Sigma_xX)

            # ReLU activation
            # Sigma_xx = torch.relu(Sigma_xx)
            # Sigma_XX = torch.relu(Sigma_XX)
            # Sigma_xX = torch.relu(Sigma_xX)

            Sigma_xX_list.append(Sigma_xX)

        # nodes_gram =  torch.mean(torch.stack(Sigma_xX_list, dim=1),dim=1)
        nodes_gram = sum(Sigma_xX_list)
        return sum(sum(nodes_gram))
