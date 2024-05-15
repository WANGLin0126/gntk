"""
python gram.py --dataset IMDBBINARY --kernel GNTK
"""

import util
import time
import numpy as np
import scipy
from os.path import join
import argparse
import os
from multiprocessing import Pool
from gntk import GNTK
from sntk import StructureBasedNeuralTangentKernel
from sgntk import SimplifyingGraphNeuralTangentKernel
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='GNTK computation')
# several folders, each folder one kernel
parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: COLLAB)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of mlp layers')
parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
parser.add_argument('--scale', type=str, default='degree',
						help='scaling methods')
parser.add_argument('--jk', type=int, default=1,
						help='whether to add jk')
parser.add_argument('--out_dir', type=str, default="out",
                    help='output directory')
parser.add_argument('--kernel', type=str, default='SGNTK', help='kernel type, [GNTK,SNTK,SGNTK]')
args = parser.parse_args()

if args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
    # social network
    degree_as_tag = True
elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
    # bioinformatics
    degree_as_tag = False
    
graphs, _  = util.load_data(args.dataset, degree_as_tag)
labels = np.array([g.label for g in graphs]).astype(int)

gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale)
A_list = []
diag_list = []

# procesing the data
for i in range(len(graphs)):
    n = len(graphs[i].neighbors)
    for j in range(n):
        graphs[i].neighbors[j].append(j)
    edges = graphs[i].g.edges
    m = len(edges)

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    A = torch.sparse_coo_tensor(torch.tensor([row, col]),torch.tensor([1] * len(edges)),size=(n, n)).to(torch.float32)

    A_list.append(A.to(device))

    shape = torch.Size([n, n])
    indices = torch.arange(0, n).unsqueeze(0).repeat(2, 1)
    values = torch.ones(n)
    sparse_eye = torch.sparse_coo_tensor(indices, values, shape).to(device)

    A_list[-1] = A_list[-1] + A_list[-1].t() + sparse_eye
    diag = gntk.diag(graphs[i], A_list[i])
    diag_list.append(diag)

SNTK = StructureBasedNeuralTangentKernel(K=args.num_layers, L=1)
SGNTK = SimplifyingGraphNeuralTangentKernel(K=args.num_layers, L=1)


if args.kernel == 'SNTK':
    def calc(T):
        return SNTK.similarity(graphs[T[0]], graphs[T[1]], A_list[T[0]], A_list[T[1]    ])
elif args.kernel == 'SGNTK':
    def calc(T):
        return SGNTK.similarity(graphs[T[0]], graphs[T[1]], A_list[T[0]], A_list[T[1]])
elif args.kernel == 'GNTK':
    def calc(T):
        return gntk.gntk(graphs[T[0]], graphs[T[1]], diag_list[T[0]], diag_list[T[1]], A_list[T[0]], A_list[T[1]])



calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

# pool = Pool(80)
# results = pool.map(calc, calc_list)
print(f"----------Calculating {args.kernel} kernel matrix----------")
results = []
# print('--------Calculating kernel results-----------')
for T in tqdm(calc_list):
    results.append(calc(T).item())

gram = torch.zeros((len(graphs), len(graphs)))
for t, v in zip(calc_list, results):
    gram[t[0], t[1]] = v
    gram[t[1], t[0]] = v
    

np.save(args.out_dir+'/'+ args.dataset+'_'+args.kernel+'_gram', gram)
np.save(args.out_dir+'/'+ args.dataset+'_'+args.kernel+'_labels', labels)
