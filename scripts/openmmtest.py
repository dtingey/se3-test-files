from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

pdb = PDBFile('water.pdb')
forcefield = ForceField('amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=5.0*angstrom, constraints=HBonds)
integrator = LangevinIntegrator(298.0,0.1,0.0005*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('/results/outputtip3pminimizedlangevin.pdb', 5))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(200_000)
