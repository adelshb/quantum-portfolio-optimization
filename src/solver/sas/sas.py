# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Simulated Annealing enhanced VQE for Portfolio Optimization."""

from typing import Optional

import numpy as np
from numpy import ndarray
from scipy import optimize

from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer

from solver.vqe.vqe_solver import VQESolver

from simanneal import Annealer

class SASolver():
        """
        Class for Simulated Annealing Solver (SAS) with VQE for Portfolio Optimization.
        """
        def __init__(self,
                Cov: ndarray,
                ) -> None:

                # Get parameters
                self._Cov = Cov
                self._N = Cov.shape[0]
                self._vqe = VQESolver()
                self._vqe.qp(Cov = self._Cov)
                self._vqe.to_ising()     

        def build_vqe_instance(self,
                ansatz: QuantumCircuit,
                opt: Optimizer,
                qi: QuantumInstance
                ) -> None:
                """
                Args:
                        ansatz: QuantumCircuit,
                        opt: Optimizer,
                        qi: QuantumInstance 
                """

                self._vqe.vqe_instance(ansatz=ansatz,
                                optimizer=opt, 
                                # init= self.params,
                                quantum_instance=qi)

        # pass extra data (the distance matrix) into the constructor
        def __init__(self, state, distance_matrix):
                self.distance_matrix = distance_matrix
                super(TravellingSalesmanProblem, self).__init__(state)  # important!

        def move(self):
                """Swaps two cities in the route."""
                # no efficiency gain, just proof of concept
                # demonstrates returning the delta energy (optional)
                initial_energy = self.energy()

                a = random.randint(0, len(self.state) - 1)
                b = random.randint(0, len(self.state) - 1)
                self.state[a], self.state[b] = self.state[b], self.state[a]

                return self.energy() - initial_energy

        def solve(self,
                params: ndarray,
                T0: Optional[float],
                schedule: Optional[str] = 'boltzmann',
                maxiter: Optional[int] = 500,
                lower: float = 0.0,
                upper: float = 2 * np.pi,
                dwell: Optional[int] = 250,
                disp: Optional[bool] = True,
                ) -> None:
                """

                """

                self.sa = optimize.basinhopping(self.f, x0=params, T=T0, niter=maxiter, disp=disp)


        def energy(self, params: ndarray, return_expectation: Optional[bool]=True) -> float:

                E = self.vqe.eval(params, return_expectation)
                return E

        @property
        def vqe(self) -> VQESolver:
                """ Returns the VQE solver. """
                return self._vqe

        @vqe.setter
        def vqe(self, value: VQESolver) -> None:
                """ Sets the VQE solver. """
                self._vqe = value