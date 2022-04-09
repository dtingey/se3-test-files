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

implicit_water = GBSAOBCForce() 
potential = MLPotential('ani2x')

#import pdb; pdb.set_trace()

#######OpenMM Stuff################

pdbf = PDBFile('trialanine.pdb')
#modeller = Modeller(pdbf.topology, pdbf.positions)
#modeller.addSolvent(forcefield, boxSize=Vec3(1.0, 1.0, 1.0)*nanometers)

#Create System and Topology
topo = pdbf.topology

mm_system = System(topo)
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)

for atom in topo.atoms():
    index = int(atom.id)-1
    impicit_water.addParticle()



chains = list(topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]

ml_system = potential.createMixedSystem(topo, mm_system, ml_atoms)



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
simulation.reporters.append(PDBReporter('trialanineoutanitest.pdb', 5))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10_000)

print()

#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

