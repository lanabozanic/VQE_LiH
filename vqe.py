import numpy as np
import pylab
import copy
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType


molecule = 'H .0 .0 -{0}; Li .0 .0 {0}'
algorithms = ['VQE', 'ExactEigensolver']

dr  = [x * 0.1  for x in range(6, 20)]
dr += [x * 0.25 for x in range(8, 18)]
dr += [4.0]
energies = np.empty([len(algorithms), len(dr)])
hf_energies = np.empty(len(dr))
distances = np.empty(len(dr))



for i, d in enumerate(dr):
    for j in range(len(algorithms)):   
        driver = PySCFDriver(molecule.format(d/2), basis='sto3g')
        qmolecule = driver.run()
        operator =  Hamiltonian(qubit_mapping=QubitMappingType.PARITY,
                                two_qubit_reduction=True, freeze_core=True,
                                orbital_reduction=[-3, -2])
        qubit_op, aux_ops = operator.run(qmolecule)
        if algorithms[j] == 'ExactEigensolver':
            result = ExactEigensolver(qubit_op, aux_operators=aux_ops).run()
        optimizer = COBYLA(maxiter=1000)
        initial_state = HartreeFock(qubit_op.num_qubits,
                                    operator.molecule_info['num_orbitals'],
                                    operator.molecule_info['num_particles'],
                                    qubit_mapping=operator._qubit_mapping,
                                    two_qubit_reduction=operator._two_qubit_reduction)
        
        var_form = UCCSD(qubit_op.num_qubits, depth=1,
                        num_orbitals=operator.molecule_info['num_orbitals'],
                        num_particles=operator.molecule_info['num_particles'],
                        initial_state=initial_state,
                        qubit_mapping=operator._qubit_mapping,
                        two_qubit_reduction=operator._two_qubit_reduction)
        
        algo = VQE(qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
            
        lines, result = operator.process_algorithm_result(result)
        energies[j][i] = result['energy']
        hf_energies[i] = result['hf_energy']


    distances[i] = d



pylab.plot(distances, hf_energies, label='Hartree-Fock')
for j in range(len(algorithm)):
    pylab.plot(distances, energies[j], label=algorithms[j])
pylab.xlabel('Interatomic distance')
pylab.ylabel('Energy')
pylab.title('LiH Ground State Energy')
pylab.legend(loc='upper right');
