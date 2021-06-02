from qiskit import BasicAer
from qiskit.aqua.operators import X, Z, I
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal


H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)


from qiskit.aqua import aqua_globals
seed = 50
aqua_globals.random_seed = seed
qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(operator=H2_op, var_form=ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.run()

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(result)


# initial_pt = result.optimal_point

# aqua_globals.random_seed = seed
# qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

# ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
# slsqp = SLSQP(maxiter=1000)
# vqe = VQE(operator=H2_op, var_form=ansatz, optimizer=slsqp, initial_point=initial_pt, quantum_instance=qi)
# result1 = vqe.run()

# pp.pprint(result1)