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
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP

import matplotlib.pyplot as plt

def main(args):

    # Cov = randcovmat(args.d)
    Cov = np.array([[1,0],[0,0]])
    # Cov = np.array([[1,0,0],[0,0,0],[0,0,0]])

    # CVXPY
    w_cvxpy = CVXPYSolver(Cov)
    print("CVXPY (MOSEK): ", w_cvxpy.T @ Cov @ w_cvxpy)

    # VQE
    data = {}
    N = Cov.shape[0]
    vqe = VQESolver()
    vqe.qp(Cov = Cov)

    # Prepare QuantumInstance
    qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=args.seed, seed_simulator=args.seed)
    
    optimizers = [COBYLA(maxiter=args.maxiter)]

    for n in range(1,args.maxNq+1):
        vqe.to_ising(Nq = n)
        data[n] = {}
        for opt in optimizers:
            data[n][type(opt).__name__] = []
            for rep in range(1,args.maxreps+1):

                counts = []
                values = []
                def store_intermediate_result(eval_count, parameters, mean, std):
                                    counts.append(eval_count)
                                    values.append(mean)

                ansatz = TwoLocal(num_qubits=N*n, 
                                    rotation_blocks=['ry','rz'], 
                                    entanglement_blocks='cz',
                                    reps=rep,
                                    entanglement='full')

                vqe.vqe_instance(ansatz=ansatz,
                                optimizer=opt, 
                                quantum_instance=qi, 
                                callback=store_intermediate_result)

                res_vqe = vqe.solve()
                # print("VQE: ", res_vqe)

                data[n][type(opt).__name__].append({"reps": rep,
                                            "counts": np.asarray(counts),
                                            "values": np.asarray(values) + vqe.offset })
    for n in range(1,args.maxNq+1):
        for opt in optimizers:
            name = type(opt).__name__
            for rep in range(1,args.maxreps+1):

                plt.plot(data[n][name][rep-1]['counts'], data[n][name][rep-1]['values'], label=name + " reps={} Nq={}".format(rep,n))

    plt.xlabel('Eval count')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    # Problem parameters
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--maxNq", type=int, default=3)

    # Quantum Solver parameters
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--maxreps", type=int, default=2)

    args = parser.parse_args()
    main(args)