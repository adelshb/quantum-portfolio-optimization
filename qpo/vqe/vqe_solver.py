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

from numpy import ndarray 

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import VQE

from .continuous_to_binary import ContinuousToBinary

class VQESolver(object):
        """
        Class for VQE Solver for Portfolio Optimization
        """

        def qp(self, Cov: ndarray)-> None:
                """
                Take a Covariance matrix (from the different assets) and minimize the risk via VQE optimization.
                """
                N = Cov.shape[0]

                self._qp = QuadraticProgram('portfolio_optimization')

                self._qp.continuous_var_list([str(i) for i in range(self._N)], lowerbound=0, upperbound=1, name="w")
                self._qp.linear_constraint(linear=[1]*self._N, sense='EQ', rhs=1, name='total investment')
                self._qp.minimize(constant=0.0, linear=None, quadratic=self._Cov)

        def to_ising(self,  Nq: int)-> None:

                # Convert continous variables to binary
                con2bin = ContinuousToBinary(Nq)
                qp_bin = con2bin.convert(self._qp)

                # Convert to QUBO then to Ising
                conv = QuadraticProgramToQubo()
                self._qubo = conv.convert(qp_bin)
                H, offset = self._qubo.to_ising()
                return H, offset

        def vqe_instance(self, ansatz, optimizer, quantum_instance):

                # # Prepare QuantumInstance
                # qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=self._seed, seed_simulator=self._seed)

                # # Select the VQE parameters
                # ansatz = TwoLocal(num_qubits=self._N*self._Nq, rotation_blocks='ry', entanglement_blocks='cz')
                # slsqp = SLSQP(maxiter=100)

                vqe = VQE(ansatz = ansatz, 
                        optimizer = optimizer, 
                        initial_point = None, 
                        gradient = None, 
                        expectation = None, 
                        include_custom = False, 
                        max_evals_grouped = 1, 
                        callback = None, 
                        quantum_instance = quantum_instance)

        def solve(self)->None:
                res = self._vqe.compute_minimum_eigenvalue(self._H)
                return  res.optimal_value + self._offset