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
""" QAOA Optimization"""

from typing import Optional

import numpy as np 

from qiskit import BasicAer
from qiskit.utils import QuantumInstance

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import LinearEqualityToPenalty
from qiskit.optimization.algorithms import MinimumEigenOptimizer

from qiskit.algorithms import QAOA

def QAOASolver(Cov: object,
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
                result : QAOA result object.
        """

        # Prepare Quadratic Program
        N = Cov.shape[0]
        K = 2**Nq - 1
        mod = QuadraticProgram('portfolio_optimization')

        # Record variable with binary encoding
        for i in range(N):
                for j in range(Nq):
                        mod.binary_var('x'+str(i)+str(j))

        # Objective function in binary encoding form
        B = np.array([2**n for n in range(Nq**2)]).reshape((Nq,Nq))
        Q = np.kron(Cov, B)/(K**2)
        mod.minimize(constant=N, quadratic=Q)

        # Sum asset allocation is 1 in binary encoding form
        const = {"x"+str(i)+str(j): 2**j for i in range(N) for j in range(Nq)}
        mod.linear_constraint(linear=const, sense='==', rhs=K**2, name='lin_eq')
        
        # Change to QUBO
        lineq2penalty = LinearEqualityToPenalty()
        qubo = lineq2penalty.convert(mod)

        # Prepare QuantumInstance
        qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

        # Select the QAOA parameters
        qaoa_mes = QAOA(quantum_instance=qi)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        qaoa_result = qaoa.solve(qubo)

        return  qaoa_result