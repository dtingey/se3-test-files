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
PARSER.add_argument('-i', dest='inFile',type=str, help='Path to the directory with the input pdb file')
PARSER.add_argument('-o', dest='outFile',type=str, default='md_output.pdb',
                      help='The name of the output file that will be written in /results/, default=md_output.pdb')
PARSER.add_argument('-s', dest='stepSize',type=float, default=0.5,
                      help='Step size in femtoseconds, default=0.5')
PARSER.add_argument('-t', dest='simTime',type=float, default=1.0,
                      help='Simulation time in picoseconds, default=1')
PARSER.add_argument('-m', dest='modelFile',type=str, default='model_ani1x_5_12.pth',
                      help='.pth model file name, default=model_ani1x_5_12.pth')

args = PARSER.parse_args()
#initialize parameters
in_file = args.inFile
out_file = args.outFile
step_size = args.stepSize
simulation_time = args.simTime
model_file = args.modelFile

######Load Model############
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
checkpoint = torch.load(f'./{model_file}', map_location=device)
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


#waterforce = GBSAOBCForce()

#import pdb; pdb.set_trace()
print('############SUCCESS##########')
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)
    #waterforce.addParticle(1,0.1,0.1)

#system.addForce(waterforce)

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


sm = SE3Module(model)

count = 0

def energy_function(positions):
    global count
    count+=1
    print(f'Energy Function called: {count}')
    positions = torch.tensor(positions, dtype=torch.float).reshape(-1,3)
    energy = sm(species, positions, forces=False)
    print(f'Energy: {energy:0.5f}')
    return energy

def jacobian(positions):
    print('Force Function called')
    positions = torch.tensor(positions, dtype=torch.float).reshape(-1,3)
    energy, forces = sm(species, positions)
    norm_forces = torch.norm(forces)
    print(f'Forces: {norm_forces:0.5f}')
    return -forces.to('cpu').flatten()




pos = torch.FloatTensor(pdbf.getPositions(asNumpy=True).tolist())*10.0
energy, forces = sm(species, pos)
norm_forces = torch.norm(forces).item()

#import pdb; pdb.set_trace()

######################### ENERGY MINIMIZATION ############################

print(f"Energy: {energy:.3f}")
print(f"Forces: {norm_forces:.3f}")

print("Minimizing energy....")

res = scipy.optimize.minimize(energy_function, pos.flatten(), method='CG', jac=jacobian)
newpos = torch.tensor(res.x, dtype=torch.float).reshape(-1,3)

energy, forces = sm(species, newpos)
print(f"Energy: {energy:.3f}")
print(f"Forces: {torch.norm(forces).item():.3f}")

####################### ADD INITIAL FORCES ##################################

#import pdb; pdb.set_trace()
index = 0
for atom in topo.atoms():
    #c = forces[index] @ newpos[index].to('cuda')
    c = 0
    se3force.addParticle(index, (forces[index][0].item()*627.5, forces[index][1].item()*627.5, forces[index][2].item()*627.5)*kilocalorie_per_mole/angstrom)
    index+=1

print("Add forces to particle Worked")

####################### ADD INITIAL VELOCITIES ###########################
kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA

def generateMaxwellBoltzmannVelocities(system, temperature):
    """Generate Maxwell-Boltzmann velocities.

    ARGUMENTS
        system (simtk.openmm.System) - the system for which velocities are to be assigned
        temperature (simtk.unit.Quantity of temperature) - the temperature at which velocities are to be assigned

    RETURNS

    velocities (simtk.unit.Quantity of numpy Nx3 array, units length/time) - particle velocities

    TODO
    This could be sped up by introducing vector operations.

    """

    # Get number of atoms
    natoms = system.getNumParticles()

    # Create storage for velocities.
    velocities = Quantity(np.zeros([natoms, 3], np.float32), nanometer / picosecond) # velocities[i,k] is the kth component of the velocity of atom i

    # Compute thermal energy and inverse temperature from specified temperature.
    kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
    temperature = 298.0 * kelvin
    kT = kB * temperature # thermal energy
    beta = 1.0 / kT # inverse temperature

    # Assign velocities from the Maxwell-Boltzmann distribution.
    for atom_index in range(natoms):
        mass = system.getParticleMass(atom_index) # atomic mass
        sigma = sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
        for k in range(3):
            velocities[atom_index,k] = sigma * np.random.normal()
    # Return velocities
    return velocities

####################### SETUP AND RUN SIMULATION #############################

newpos = pos
#Create Integrator and Simulation
temperature = 298.0 * kelvin
#integrator = VerletIntegrator(step_size*femtosecond)
integrator = LangevinIntegrator(temperature, 0.1/femtosecond, step_size*femtosecond)

#Simulation?
simulation = Simulation(topo, system, integrator)


positions = newpos.tolist()*angstrom 
simulation.context.setPositions(positions)
velocities = generateMaxwellBoltzmannVelocities(system, temperature)
simulation.context.setVelocities(velocities)
#simulation.context.setPeriodicBoxVectors([1.0,0,0],[0,1.0,0],[0,0,1.0])


#Print Initial Positions
print('############Before###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
#import pdb; pdb.set_trace()
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
    newpos = torch.FloatTensor([[pos.x,pos.y,pos.z] for pos in positions])*10.0 # Nanometer to Angstrom
    energy, forces = sm(species, newpos)
    index = 0
    for atom in topo.atoms():
        #c = forces[index] @ newpos[index].to('cuda')
        c = 0
        se3force.setParticleParameters(index, index, (forces[index][0].item()*627.5, forces[index][1].item()*627.5, forces[index][2].item()*627.5)*kilocalorie_per_mole/angstrom)
        index+=1
    se3force.updateParametersInContext(simulation.context)


#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

