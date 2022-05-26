from se3_transformer.data_loading import ANI1xDataModule
from se3_transformer.data_loading.ani1x import ANI1xDataset
from torch.nn.parallel import DistributedDataParallel
from se3_transformer.model import SE3TransformerANI1x
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.utils import using_tensor_cores, init_distributed, increase_l2_fetch_granularity
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.utils import to_cuda
from se3_transformer.runtime.training import *

import argparse
import os
import re
import random
import torch
import numpy as np
from torch import Tensor
import scipy
import dgl
import dgl.data
from dgl import DGLGraph
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmtorch import TorchForce

random.seed()

######Load Model############
args = PARSER.parse_args()
args.norm = True
args.use_layer_norm = True
args.amp = True
args.num_degrees = 3
args.cutoff = 3.0
args.channels_div = 4
model = SE3TransformerANI1x(
        fiber_in=Fiber({0: 4}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: args.num_basis_fns}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args)
    )


device = 'cuda:0' #torch.cuda.current_device()
model.to(device=device)
checkpoint = torch.load('./model_ani1x_5_12.pth', map_location=device)
#checkpoint = torch.load(f'./{model_file}', map_location={'cuda:0': f'cuda:{get_local_rank()}'})
model.load_state_dict(checkpoint['state_dict'])

ENERGY_STD = 0.1062

class SE3Module(torch.nn.Module):
    def __init__(self, trained_model):
        super(SE3Module, self).__init__() 
        self.model = trained_model
        eye = torch.eye(4)
        self.species_dict = {'H': eye[0], 'C': eye[1], 'N': eye[2], 'O': eye[3]}
    def forward(self, species, positions, forces=True): 
        species_tensor = torch.stack([self.species_dict[atom] for atom in species])
        species_tensor = to_cuda(species_tensor)
        graph = ANI1xDataset._create_graph(positions, cutoff=args.cutoff) # Cutoff for 5-12 model is 3.0 A
        graph = to_cuda(graph)
        node_feats = {'0': species_tensor.unsqueeze(-1)}
        inputs = [graph, node_feats]
        if forces:
            energy, forces = self.model(inputs, forces=forces, create_graph=False)
            return (energy*ENERGY_STD).item(), forces*ENERGY_STD
        else:
            energy = self.model(inputs, forces=forces, create_graph=False)
            return (energy*ENERGY_STD).item()



species = ['H','H']
sm = SE3Module(model)

r_array = np.linspace(0,2.9,30)
e_array = np.zeros_like(r_array)

for i, r in enumerate(r_array):
    pos = torch.FloatTensor([[0,0,0],[r,0,0]])
    energy = sm(species, pos, forces=False)
    e_array[i] = float(energy)
    print(f'Step {i}: Energy: {float(energy)}')

data = np.array([r_array, e_array])
np.save('./data.npy',data)

