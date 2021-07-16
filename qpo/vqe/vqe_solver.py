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

import numpy as np 

from qiskit import BasicAer
from qiskit.utils import QuantumInstance

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal

from .continuous_to_binary import ContinuousToBinary

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

        # Prepare Quadratic Program
        N = Cov.shape[0]
        qp = QuadraticProgram('portfolio_optimization')

        qp.continuous_var_list([str(i) for i in range(N)], lowerbound=0, upperbound=1, name="w")
        qp.linear_constraint(linear=[1]*N, sense='EQ', rhs=1, name='total investment')
        qp.minimize(constant=0.0, linear=None, quadratic=Cov)

        # Convert continous variables to binary
        con2bin = ContinuousToBinary(Nq)
        qp_bin = con2bin.convert(qp)

        # Convert to QUBO then to Ising
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp_bin)
        H, offset = qubo.to_ising()

        # Prepare QuantumInstance
        qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        # Select the VQE parameters
        ansatz = TwoLocal(num_qubits=N*Nq, rotation_blocks='ry', entanglement_blocks='cz')
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

        res = vqe.compute_minimum_eigenvalue(H)

        return  res.optimal_value + offset