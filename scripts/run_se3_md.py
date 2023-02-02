from trip.data_loading import GraphConstructor
from trip.model import TrIP
from se3_transformer.runtime.utils import to_cuda
from trip.runtime.training import *

import argparse
import os
import random
import time
from datetime import datetime
import torch
import numpy as np
from torch import Tensor
from scipy.optimize import minimize
import dgl
import dgl.data
from dgl import DGLGraph
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout


random.seed()

now = datetime.now()

parser = argparse.ArgumentParser(description='run md simulations using SE3 Forcefield')
parser.add_argument('-i', dest='inFile',type=str, help='Path to the directory with the input pdb file')
parser.add_argument('-o', dest='outFile',type=str, default='md_output.pdb',
                      help='The name of the output file that will be written in /results/, default=md_output.pdb')
parser.add_argument('-s', dest='stepSize',type=float, default=0.5,
                      help='Step size in femtoseconds, default=0.5')
parser.add_argument('-t', dest='simTime',type=float, default=1.0,
                      help='Simulation time in picoseconds, default=1')
parser.add_argument('-m', dest='modelFile',type=str, default='9-20-22.pth',
                      help='.pth model file name, default=')

args = parser.parse_args()
#initialize parameters
in_file = args.inFile
out_file = args.outFile
step_size = args.stepSize
simulation_time = args.simTime
model_file = args.modelFile


############## File Stuff ##############
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
path = os.path.join("/results/", dt_string)

os.mkdir(path)


def calculate_time(start, end, directory_path=path, message="Unknown"):
    file_path = os.path.join(directory_path, "time_analysis.txt")
    with open(file_path, "a") as f:
        f.write(f"{message}\n")
        f.write(f"Start time: {start}\n")
        f.write(f"End time: {end}\n")
        f.write(f"Time Elapsed: {end - start} seconds\n\n")



start = time.time()


######Load Model############
model = TrIP.load(f'/results/{model_file}', map_location='cuda:0')

ENERGY_STD = 1.0

class SE3Module(torch.nn.Module):
    def __init__(self, trained_model, species):
        super(SE3Module, self).__init__() 
        self.model = trained_model
        eye = torch.eye(4)
        self.species_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
        self.graph_constructor = GraphConstructor(cutoff=trained_model.cutoff)
        self.species_tensor = torch.tensor([self.species_dict[atom] for atom in species], dtype=torch.int, device='cuda')
	
    def forward(self, positions, forces=True):
        time1 = time.time()
        graph = self.graph_constructor.create_graphs(positions, torch.tensor(float('inf'))) # Cutoff for 5-12 model is 3.0 A
        time2= time.time()
        calculate_time(time1, time2, message="Graph Creation")
        
        graph.ndata['species'] = self.species_tensor
        

        time1 = time.time()
        if forces:
            energy, forces = self.model(graph, forces=forces, create_graph=False)
            time2 = time.time()
            calculate_time(time1, time2, message="Energy and Force Calculation:")
            return (energy*ENERGY_STD).item(), forces*ENERGY_STD
        else:
            energy = self.model(graph, forces=forces, create_graph=False)
            time2 = time.time()
            calculate_time(time1, time2, message="Energy Calculation")
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


sm = SE3Module(model, species)

count = 0

def energy_function(positions):
    global count
    count+=1
    print(f'Energy Function called: {count}')
    positions = torch.tensor(positions, dtype=torch.float, device='cuda').reshape(-1,3)
    energy = sm(positions, forces=False)
    print(f'Energy: {energy:0.5f}')
    return energy

def jacobian(positions):
    print('Force Function called')
    positions = torch.tensor(positions, dtype=torch.float, device='cuda').reshape(-1,3)
    energy, forces = sm(positions)
    norm_forces = torch.norm(forces)
    print(f'Forces: {norm_forces:0.5f}')
    return -forces.detach().cpu().numpy().flatten()



pos = torch.tensor(pdbf.getPositions(asNumpy=True)/angstrom, dtype=torch.float, device='cuda')
energy, forces = sm(pos)
norm_forces = torch.norm(forces).item()

#import pdb; pdb.set_trace()



######################### ENERGY MINIMIZATION ############################

print(f"Energy: {energy:.3f}")
print(f"Forces: {norm_forces:.3f}")

print("Minimizing energy....")

time1 = time.time()
res = minimize(energy_function, pos.cpu().numpy().flatten(), method='CG', jac=jacobian)
newpos = torch.tensor(res.x, dtype=torch.float, device='cuda').reshape(-1,3)

energy, forces = sm(newpos)
print(f"Energy: {energy:.3f}")
print(f"Forces: {torch.norm(forces).item():.3f}")
time2 = time.time()

calculate_time(time1, time2, path, "Minimization Time")


####################### ADD INITIAL FORCES ##################################

#import pdb; pdb.set_trace()


time1 = time.time()
index = 0
for atom in topo.atoms():
    #c = forces[index] @ newpos[index].to('cuda')
    c = 0
    se3force.addParticle(index, (forces[index][0].item()*627.5, forces[index][1].item()*627.5, forces[index][2].item()*627.5)*kilocalorie_per_mole/angstrom)
    index+=1

print("Add forces to particle Worked")
time2 = time.time()
calculate_time(time1, time2, path, "Initial Forces")


time1 = time.time()
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

time2 = time.time()
calculate_time(time1, time2, path, "Initialize Velocities")



####################### SETUP AND RUN SIMULATION #############################
time1 = time.time()
newpos = pos
#Create Integrator and Simulation
temperature = 298.0 * kelvin
#integrator = VerletIntegrator(step_size*femtosecond)
integrator = LangevinIntegrator(temperature, 1/picosecond, step_size*femtosecond)

#Simulation?
platform = Platform.getPlatformByName('CUDA')
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
simulation.reporters.append(PDBReporter(f'{os.path.join(path, out_file)}', 3))
simulation.reporters.append(StateDataReporter(stdout, 100, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))

num_steps = int((simulation_time/step_size)*1000)

time2 = time.time()
calculate_time(time1, time2, path, "Simulation Setup:")


for i in range(num_steps):
    simulation.step(1)
    #simulation.topology.createStandardBonds()
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    newpos = torch.tensor([[pos.x,pos.y,pos.z] for pos in positions], dtype=torch.float, device='cuda')*10.0 # Nanometer to Angstrom
    
    time1 = time.time()
    energy, forces = sm(newpos)
    time2 = time.time()
    calculate_time(time1, time2, path, "Energy from Model")
    
    forces *= 627.5
    forces = forces*kilocalorie_per_mole/angstrom
    
    #import pdb; pdb.set_trace()

    time1 = time.time()
    for index, atom in enumerate(topo.atoms()):
        #c = forces[index] @ newpos[index].to('cuda')
        c = 0
        se3force.setParticleParameters(index, index, forces[index])
        
    se3force.updateParametersInContext(simulation.context)
    time2 = time.time()
    calculate_time(time1, time2, path, "Updating Forces")

#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)


end = time.time()
calculate_time(start, end, path, "Total Time Elapsed")

