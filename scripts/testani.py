import re
import random
import torch
import numpy as np
from torch import Tensor
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmtorch import TorchForce
from sys import stdout
import torchani
import openmmml
from openmmml import MLPotential
#import pdb; pdb.set_trace()

potential = MLPotential('ani2x')

#import pdb; pdb.set_trace()

#######OpenMM Stuff################

pdbf = PDBFile('water.pdb')

#Create System and Topology
topo = pdbf.topology

system = potential.createSystem(topo, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=0.5*angstrom)

#import pdb; pdb.set_trace()

'''
print('############SUCCESS##########')
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)


system.addForce(torch_force)

#Create Custom Force 

aniforce = CustomExternalForce('-fx*x-fy*y-fz*z')
system.addForce(aniforce)
aniforce.addPerParticleParameter('fx')
aniforce.addPerParticleParameter('fy')
aniforce.addPerParticleParameter('fz')
print("Force Worked")

species = []
for atom in topo.atoms():
    sym = atom.element.symbol
    species.append(sym)

pos = torch.FloatTensor(pdbf.getPositions(asNumpy=True).tolist())*10.0
ani = ANIModule(model)
energy, forces = ani(species, pos)

import pdb; pdb.set_trace()

for atom in topo.atoms():
    index = int(atom.id)-1
    aniforce.addParticle(index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item())*kilocalorie_per_mole/angstrom)


print("Add forces to particle Worked")
'''
#Create Integrator and Simulation
integrator = LangevinIntegrator(298.0,0.02/femtosecond,0.5*femtosecond)

#Simulation?
simulation = Simulation(topo, system, integrator)

positions = pdbf.getPositions()
simulation.context.setPositions(positions)
simulation.minimizeEnergy()

print()

#Run simulation and do force calculations
simulation.reporters.append(PDBReporter('wateroutanitest.pdb', 5))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10_000)

print()

#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

