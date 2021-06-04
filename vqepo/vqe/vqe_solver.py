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
""" VQE Optimization"""

from typing import Optional

from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal

from vqepo.vqe.qubo import bin_enc, qubo2ising

def VQESolver(Cov: object,
                Nq: int,
                #backend: Optional[str] = "statevector",
                seed: Optional[int] = None
                )-> object :
        """
        Take a Covariance matrix (from the different assets) and minimize the risk via VQE optimization.

        Args:
                Cov : Covariance matrix
                Nq : Each variable is encoded on Nq bits.
                backend : Backend for qiskit QuantumInstance.
                seed : seed for QuantumInstance.

        Returns:
                result : VQE result object.
        """

        # Compute the Hamiltonian via binarization encoding and construct the Pauli Operators
        Q = bin_enc(Nq, Cov)
        H = qubo2ising(Q)

        # Prepare QuantumInstance
        qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        # Select the VQE parameters
        ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
        slsqp = SLSQP(maxiter=100)

        vqe = VQE(ansatz = ansatz, 
                optimizer = slsqp, 
                initial_point = None, 
                gradient = None, 
                expectation = None, 
                include_custom = False, 
                max_evals_grouped = 1, 
                callback = None, 
                quantum_instance = qi)

        result = vqe.compute_minimum_eigenvalue(operator = H)

        return result