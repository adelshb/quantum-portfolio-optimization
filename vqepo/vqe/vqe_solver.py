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

from vqepo.vqe.qubo import bin_enc

def VQESolver(Cov: object,
                Nq: int,
                #backend: Optional[str] = "statevector",
                seed: Optional[int] = None
                )-> object :
        """
        Take a Covariance matrix (from the different assets) and minimize the risk via VQE optimization.

        Input
        ----------
        Cov : Covariance matrix
        Nq : Each variable is encoded on Nq bits.
        backend : Backend for qiskit QuantumInstance.
        seed : seed for QuantumInstance.

        Output
        ----------
        result : VQE result object.
        """

        # Compute the Hamiltonian via binarization encoding
        H = bin_enc(Nq, Cov)

        # Prepare QuantumInstance
        qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        # Select the VQE parameters
        ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
        slsqp = SLSQP(maxiter=100)
        vqe = VQE(operator=H, var_form=ansatz, optimizer=slsqp, quantum_instance=qi)
        result = vqe.run()

        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(result)

        return result