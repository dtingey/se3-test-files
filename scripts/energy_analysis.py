from trip.data_loading import TrIPDataModule, GraphConstructor
from torch.nn.parallel import DistributedDataParallel
from trip.model import TrIP
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.utils import using_tensor_cores, init_distributed, increase_l2_fetch_granularity
from trip.runtime.arguments import PARSER
from se3_transformer.runtime.utils import to_cuda
from trip.runtime.training import *

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

random.seed()
args = PARSER.parse_args()
#initialize parameters

######Load Model############
args.norm = True
args.use_layer_norm = True
args.amp = True
args.num_degrees = 3
args.cutoff = 4.6
args.channels_div = 2
args.num_channels = 16
model = TrIP(
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args)
    )


device = 'cuda:0' #torch.cuda.current_device()
model.to(device=device)
checkpoint = torch.load('./9_20_22.pth', map_location=device)
#checkpoint = torch.load(f'./{model_file}', map_location={'cuda:0': f'cuda:{get_local_rank()}'})
model.load_state_dict(checkpoint['state_dict'])

ENERGY_STD = 1

class SE3Module(torch.nn.Module):
    def __init__(self, trained_model):
        super(SE3Module, self).__init__() 
        self.model = trained_model
        eye = torch.eye(4)
        self.species_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
        self.graph_constructor = GraphConstructor(cutoff=args.cutoff)
    def forward(self, species, positions, forces=True):
        species_tensor = torch.tensor([self.species_dict[atom] for atom in species], dtype=torch.int)
        species_tensor, positions = to_cuda([species_tensor, positions])
        graph = self.graph_constructor.create_graphs(positions, torch.tensor(float('inf'))) # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = species_tensor
        if forces:
            energy, forces = self.model(graph, forces=forces, create_graph=False)
            return (energy*ENERGY_STD).item(), forces*ENERGY_STD
        else:
            energy = self.model(graph, forces=forces, create_graph=False)
            return (energy*ENERGY_STD).item()


for symbol, name in zip(['H','C','N','O'],['hydrogen','carbon','nitrogen','oxygen']):
    species = [symbol, symbol]
    sm = SE3Module(model)

    r_array = np.linspace(0,2.9,30)
    e_array = np.zeros_like(r_array)

    for i, r in enumerate(r_array):
        pos = torch.FloatTensor([[0,0,0],[r,0,0]])
        energy = sm(species, pos, forces=False)
        e_array[i] = float(energy)
        print(f'Step {i}: Energy: {float(energy)}')

    data = np.array([r_array, e_array])
    np.save(f'/results/data_{name}.npy',data)


