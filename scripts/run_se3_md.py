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

#parser = argparse.ArgumentParser(description='run md simulations using SE3 Forcefield')
PARSER.add_argument('-i', dest='inFile',type=str, help='path to the directory with the input pdb file')
PARSER.add_argument('-o', dest='outFile',type=str, default='md_output.pdb',
                      help='the name of the output file that will be written in /results/, default=md_output.pdb')
PARSER.add_argument('-s', dest='stepSize',type=float, default=0.5,
                      help='stepsize in femtoseconds, default=0.5')
PARSER.add_argument('-t', dest='simTime',type=float, default=1.0,
                      help='simulation time in picoseconds, default=1')

args = PARSER.parse_args()
#initialize default parameters
in_file = args.inFile
out_file = args.outFile
step_size = args.stepSize
simulation_time = args.simTime

######Load Model############
#args = PARSER.parse_args()
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

ENERGY_STD = 0.1062

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
        energy, forces = self.model(inputs, forces=True, create_graph=False)
        return energy*ENERGY_STD, forces*ENERGY_STD



#######OpenMM Stuff################

pdbf = PDBFile(in_file)

#Create System and Topology
system = System()
topo = pdbf.topology

'''
#Make a Box 1 nm
box = system.getDefaultPeriodicBoxVectors()
for i in range(3):
    box[i]=box[i]*0.5
'''


waterforce = GBSAOBCForce()

#import pdb; pdb.set_trace()
print('############SUCCESS##########')
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)
    waterforce.addParticle(1,1,1)

system.addForce(waterforce)

#Create Custom Force 

se3force = CustomExternalForce('-fx*x-fy*y-fz*z+c')
system.addForce(se3force)
se3force.addPerParticleParameter('fx')
se3force.addPerParticleParameter('fy')
se3force.addPerParticleParameter('fz')
se3force.addPerParticleParameter('c')
print("Force Worked")

species = []
for atom in topo.atoms():
    sym = atom.element.symbol
    species.append(sym)


sm = SE3Module(model)

count = 0

def energy_function(positions):
    positions = torch.tensor(positions, dtype=torch.float).reshape(-1,3)
    energy, forces = sm(species, positions)
    global count
    count+=1
    print(f'Energy Function called: {count}')
    print(f'Energy: {energy:0.5f}')
    return energy.item()

def jacobian(positions):
    positions = torch.tensor(positions, dtype=torch.float).reshape(-1,3)
    energy, forces = sm(species, positions)
    norm_forces = torch.norm(forces)
    print('Force Function called')
    print(f'Forces: {norm_forces:0.5f}')
    return -forces.to('cpu').flatten()




pos = torch.FloatTensor(pdbf.getPositions(asNumpy=True).tolist())*10.0
energy, forces = sm(species, pos)
norm_forces = torch.norm(forces)
print(f"Energy: {energy:.3f}")
print(f"Forces: {norm_forces:.3f}")


print("Minimizing energy....")

res = scipy.optimize.minimize(energy_function, pos.flatten(), method='CG', jac=jacobian)
newpos = torch.tensor(res.x, dtype=torch.float).reshape(-1,3)

energy, forces = sm(species, newpos)
print(f"Energy: {energy:.3f}")
print(f"Forces: {torch.norm(forces):.3f}")


'''
#Version 2 Energy Minimization
MAX_COUNT = 700
STEP_SIZE = 0.05
FTOL = 2.22E-16
def minimize_energy(pos,oldenergy,oldforces,count):
    if (count < MAX_COUNT):
        newpos = pos + STEP_SIZE*oldforces.to('cpu')

        energy, forces = sm(species, newpos)
        norm_forces = torch.norm(forces)
        print(f"Energy: {energy:f}")
        print(f"Forces: {norm_forces:f}")
        if (((oldenergy-energy)/max(abs(oldenergy),abs(energy),1)) > FTOL):
            return minimize_energy(newpos,energy,forces,count+1)
        else:
            return newpos, energy, forces
    else:
        return pos, oldenergy, oldforces


newpos, energy, forces = minimize_energy(pos, energy, forces, 0)

norm_forces = torch.norm(forces)
print("FINAL FORCES")
print(f"Energy: {energy:.3f}")
print(f"Forces: {norm_forces:.3f}")

import pdb; pdb.set_trace()

'''
'''
#Version 1 energy minimization step
newpos = pos
change_ratio = 1
step_size = 0.05
while norm_forces > 0.09:
    #newpos = pos + step_size(forces)
    newpos = newpos + step_size*forces.to('cpu')
    
    #se3-transformer(newpos)
    energy, forces = sm(species, newpos)
    norm_forces = torch.norm(forces)
    print(f"Energy: {energy:.3f}")
    print(f"Forces: {norm_forces:.3f}")
'''


for atom in topo.atoms():
    index = int(atom.id)-1
    #c = forces[index] @ newpos[index].to('cuda')
    c = 0
    se3force.addParticle(index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item(), c)*kilocalorie_per_mole/angstrom)


print("Add forces to particle Worked")


#Create Integrator and Simulation
integrator = LangevinIntegrator(298.0,0.02/femtosecond,step_size*femtosecond)

#Simulation?
simulation = Simulation(topo, system, integrator)


positions = newpos.tolist()*angstrom 
simulation.context.setPositions(positions)
#simulation.context.setPeriodicBoxVectors([1.0,0,0],[0,1.0,0],[0,0,1.0])


#Print Initial Positions
print('############Before###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

print()

#Run simulation and do force calculations
simulation.reporters.append(PDBReporter(f'/results/{out_file}', 5))
simulation.reporters.append(StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))

num_steps = int((simulation_time/step_size)*1000)

for i in range(num_steps):
    simulation.step(1)
    #simulation.topology.createStandardBonds()
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    newpos = torch.FloatTensor([[pos.x,pos.y,pos.z] for pos in positions])*10.0
    energy, forces = sm(species, newpos)
    for atom in topo.atoms():
        index = int(atom.id)-1
        #c = forces[index] @ newpos[index].to('cuda')
        c = 0
        se3force.setParticleParameters(index, index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item(), c)*kilocalorie_per_mole/angstrom)

    se3force.updateParametersInContext(simulation.context)


#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

