# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Optional

from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal

def vqe_solver(H: object,
                backend: Optional[str] = "statevector",
                seed: Optional[int] = None
                )-> float :


        qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
        slsqp = SLSQP(maxiter=1000)
        vqe = VQE(operator=H, var_form=ansatz, optimizer=slsqp, quantum_instance=qi)
        result = vqe.run()

        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(result)

        return result