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

import re
import random
import torch
import numpy as np
from torch import Tensor
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
model = SE3TransformerANI1x(
        fiber_in=Fiber({0: 4}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: 0}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args)
    )


device = torch.cuda.current_device()
model.to(device=device)
checkpoint = torch.load('./model_ani1x_3-14.pth', map_location={'cuda:0': f'cuda:{get_local_rank()}'})
model.load_state_dict(checkpoint['state_dict'])

class SE3Module(torch.nn.Module):
    def __init__(self, trained_model):
        super(SE3Module, self).__init__() 
        self.model = trained_model
        eye = torch.eye(4)
        self.species_dict = {'H': eye[0], 'C': eye[1], 'N': eye[2], 'O': eye[3]}
    def forward(self, species, positions): 
        species_tensor = torch.stack([self.species_dict[atom] for atom in species])
        species_tensor = to_cuda(species_tensor)
        graph = ANI1xDataset._create_graph(positions, cutoff=5.0) # Cutoff for 3-14 model is 5.0 A
        graph = to_cuda(graph)
        node_feats = {'0': species_tensor.unsqueeze(-1)}
        inputs = [graph, node_feats]
        return self.model(inputs, forces=True, create_graph=False)


#######OpenMM Stuff################

pdbf = PDBFile('ethyleneglycol.pdb')

#Create System and Topology
system = System()
topo = pdbf.topology


print('############SUCCESS##########')
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)


#Create Custom Force 

se3force = CustomExternalForce('-fx*x-fy*y-fz*z')
system.addForce(se3force)
se3force.addPerParticleParameter('fx')
se3force.addPerParticleParameter('fy')
se3force.addPerParticleParameter('fz')
print("Force Worked")

species = []
for atom in topo.atoms():
    sym = atom.element.symbol
    species.append(sym)

pos = torch.FloatTensor(pdbf.getPositions(asNumpy=True).tolist())*10.0
sm = SE3Module(model)
energy, forces = sm(species, pos)

for atom in topo.atoms():
    index = int(atom.id)-1
    se3force.addParticle(index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item())*kilocalorie_per_mole/angstrom)


print("Add forces to particle Worked")

#Create Integrator and Simulation
integrator = VerletIntegrator(0.0005)

#Simulation?
simulation = Simulation(topo, system, integrator)

positions = pdbf.getPositions()
simulation.context.setPositions(positions)

#Print Initial Positions
print('############Before###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

print()

#Run simulation and do force calculations
simulation.reporters.append(PDBReporter('/results/egoutput.pdb', 10))

for i in range(200_000):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    newpos = torch.FloatTensor([[pos.x,pos.y,pos.z] for pos in positions])*10.0
    energy, forces = sm(species, newpos)
    for atom in topo.atoms():
        index = int(atom.id)-1
        se3force.setParticleParameters(index, index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item())*kilocalorie_per_mole/angstrom)

    se3force.updateParametersInContext(simulation.context)


#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

