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

""" Iterated local search enhanced VQE for Portfolio Optimization."""

from typing import Optional, List

import numpy as np
from numpy import ndarray
from scipy.stats import qmc
from tqdm import tqdm

from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer

from qpo.vqe.vqe_solver import VQESolver
from utils import new_inter

class ILSSolver():
        """
        Class for Iterated local search (ILS) Solver with VQE for Portfolio Optimization.
        """
        def __init__(self,
                Cov: ndarray,
                sampler_method: Optional[str] = "sobol",
                ) -> None:

                # Get parameters
                self._Cov = Cov
                self._N = Cov.shape[0]
                self._vqe = VQESolver()
                self._vqe.qp(Cov = self._Cov)
                self._vqe.to_ising()

                self._sampler_method = sampler_method

        def compute_seq(self, N: int,
                        num_parameters: int) -> None:
                """Generate a sequence of input ansatz variables
                Args:
                        N: Length of the sequence.
                """
                if self._sampler_method == "sobol":                        
                        samples = qmc.Sobol(d=num_parameters, scramble=False).random_base2(m=int(np.log2(N)))
                else:
                        samples = np.random.uniform(low=0, high=1, size=(N, num_parameters))

                self.samples = samples

        def solve(self,
                ansatz: QuantumCircuit,
                opt: Optimizer,
                qi: QuantumInstance
                ) -> List:
                """
                Args:
                        ansatz: QuantumCircuit,
                        opt: Optimizer,
                        qi: QuantumInstance 
                """

                data = []  
                for samp in tqdm(self.samples):

                        init_weights = new_inter(samp)

                        counts = []
                        values = []
                        param = []
                        def store_intermediate_result(eval_count, parameters, mean, std):
                                        counts.append(eval_count)
                                        values.append(mean)
                                        param.append(parameters)

                        self._vqe.vqe_instance(ansatz=ansatz,
                                        optimizer=opt, 
                                        init= init_weights,
                                        quantum_instance=qi, 
                                        callback=store_intermediate_result)

                        self._vqe.solve()
                        data.append({
                                "counts": np.asarray(counts),
                                "values": np.asarray(values) + self._vqe.offset,
                                "parameters": param
                                })

                        self._data = data
                return data

        @property
        def vqe(self) -> VQESolver:
                """ Returns the VQE solver. """
                return self._vqe

        @vqe.setter
        def vqe(self, value: VQESolver) -> None:
                """ Sets the VQE solver. """
                self._vqe = value
