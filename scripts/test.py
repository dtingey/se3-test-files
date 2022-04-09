from se3_transformer.data_loading import ANI1xDataModule
from se3_transformer.model import SE3TransformerPooled
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime.utils import using_tensor_cores
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.utils import to_cuda

import re
import random
import torch
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

'''
######Load Model############
args = PARSER.parse_args()
args.norm = True
args.use_layer_norm = True
datamodule = QM9DataModule(task='U0',batch_size=240,num_workers=6,num_degrees=4,data_dir='./data')
model1 = SE3TransformerPooled(
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
        output_dim=1,
        tensor_cores=using_tensor_cores(False),  # use Tensor Cores more effectively,
        **vars(args)
    )

checkpoint = torch.load('./model_qm9.pth', map_location='cuda:0')
model1.load_state_dict(checkpoint['state_dict'])
model1 = to_cuda(model1)


#######Single Molecule into Model#############
dm = QM9DataModule(task='U0',batch_size=1,num_workers=6,num_degrees=4,data_dir='./data')
dl = dm.train_dataloader()

for data in dl: break

*inputs1, target1 = to_cuda(tuple(data))
#pred1 = model1(*inputs1)

def _get_relative_pos(qm9_graph: DGLGraph) -> Tensor:
    x = qm9_graph.ndata['pos']
    src, dst = qm9_graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

class SE3Module(torch.nn.Module):
    def forward(self, model, positions, graph, node_feats, edge_feats):
        graph.ndata['pos'] = positions.clone().detach()
        graph.edata['rel_pos'] = _get_relative_pos(graph)
        inputs = graph, node_feats, edge_feats
        return model(*inputs).item()

class ForceModule(torch.nn.Module):
    """A central harmonic potential that computes both energy and forces."""
    def forward(self, positions):
        """The forward method returns the energy and forces computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        forces : torch.Tensor with shape (nparticles,3)
           The force (in kJ/mol/nm) on each particle
        """
        return (torch.sum(positions**2), -2*positions)


graph, node_feats, edge_feats, *basis = inputs1
pos = graph.ndata['pos']

energy = SE3Module()
e = energy(model1, pos, graph, node_feats, edge_feats)
print(e)

'''
#######OpenMM Stuff################

xyzfile = open('somewater.xyz', 'r')

lines = xyzfile.readlines()

positions = []
for line in lines[2:]:
    l = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    l = [float(nm)*10 for nm in l]
    positions.append(l)


#Create System and Topology
system = System()
topo = Topology()

oxygen = Element.getByAtomicNumber(8)
hydrogen = Element.getByAtomicNumber(1)
o_name = oxygen.symbol
h_name = hydrogen.symbol


for molecule in range(int(len(positions)/3)):
    chain = topo.addChain()
    res = topo.addResidue('water', chain)
    topo.addAtom(o_name, oxygen, res)
    topo.addAtom(h_name, hydrogen, res)
    topo.addAtom(h_name, hydrogen, res)




print('############SUCCESS##########')
for atom in topo.atoms():
    print(atom.name + ': ' + str(atom.element.atomic_number) + ' id: ' + atom.id)
    system.addParticle(atom.element.mass)


'''
#Create Custom Force 

se3force = CustomExternalForce('-fx*x-fy*y-fz*z')
system.addForce(se3force)
se3force.addPerParticleParameter('fx')
se3force.addPerParticleParameter('fy')
se3force.addPerParticleParameter('fz')
print("Force Worked")

fm = ForceModule()
potential, forces = fm(pos)

for atom in topo.atoms():
    index = int(atom.id)-1
    se3force.addParticle(index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item())*kilocalorie_per_mole/angstrom)


print("Add forces to particle Worked")
'''
#Create Integrator and Simluation
integrator = VerletIntegrator(0.002)

#Simulation?
simulation = Simulation(topo, system, integrator)

positions = positions*angstrom
simulation.context.setPositions(positions)

#Print Initial Positions
print('############Before###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

print()

#Run simulation and do force calculations
simulation.reporters.append(PDBReporter('/results/output.pdb', 5))
simulation.step(6)
'''
for i in range(1000):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    newpos = torch.FloatTensor([[pos.x,pos.y,pos.z] for pos in positions])
    potential, forces = fm(newpos)
    for atom in res1.atoms():
        index = int(atom.id)-1
        se3force.setParticleParameters(index, index, (forces[index][0].item(), forces[index][1].item(), forces[index][2].item())*kilocalorie_per_mole/angstrom)

    se3force.updateParametersInContext(simulation.context)

'''
#Print Final Positions
print('############After###############')
state = simulation.context.getState(getPositions=True, getVelocities=True)
for position in state.getPositions():
    print(position)

