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
"""
Benchmark script
"""

from argparse import ArgumentParser

from utils import randcovmat
import numpy as np

from qpo.cvxpy.cvxpy_solver import CVXPYSolver

from qpo.vqe.vqe_solver import VQESolver

from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP

def main(args):

    # Cov = randcovmat(args.d)
    # Cov = np.array([[1,0],[0,0]])
    Cov = np.array([[1,0,0],[0,0,0],[0,0,0]])

    # CVXPY
    w_cvxpy = CVXPYSolver(Cov)
    print("CVXPY: ", w_cvxpy.T @ Cov @ w_cvxpy)

    # VQE
    vqe = VQESolver()
    vqe.qp(Cov = Cov)
    vqe.to_ising(Nq = args.Nq)

    # Prepare QuantumInstance
    qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=args.seed, seed_simulator=args.seed)

    # Select the VQE parameters
    N = Cov.shape[0]

    ansatz = TwoLocal(num_qubits=N*args.Nq, 
                        rotation_blocks=['ry','rz'], 
                        entanglement_blocks='cz',
                        reps=args.reps,
                        entanglement='full')

    slsqp = SLSQP(maxiter=args.maxiter)
    vqe.vqe_instance(ansatz=ansatz, optimizer=slsqp, quantum_instance=qi)

    res_vqe = vqe.solve()
    print("VQE: ",res_vqe)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Problem parameters
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--Nq", type=int, default=2)

    # Quantum Solver parameters
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=3)

    args = parser.parse_args()
    main(args)