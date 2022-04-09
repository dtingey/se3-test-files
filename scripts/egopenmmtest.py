from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

pdb = PDBFile('ethyleneglycol.pdb')
forcefield = ForceField('charmm36.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=5.0*angstrom, constraints=HBonds)
integrator = VerletIntegrator(0.0005*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(PDBReporter('/results/outputtip3p.pdb', 10))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10_000)
