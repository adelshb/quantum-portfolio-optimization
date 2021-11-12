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

""" Portfolio Optimization Methods Benchmark. """

from argparse import ArgumentParser

from utils import randcovmat
import numpy as np

from qpo.cvxpy.cvxpy_solver import CVXPYSolver
from qpo.ils.ils import ILSSolver

from qiskit import Aer
from qiskit.providers.aer import AerError
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA

import matplotlib.pyplot as plt

def main(args):

    Cov = randcovmat(args.d)

    # CVXPY
    w_cvxpy = CVXPYSolver(Cov, verbose = False)
    mosek = w_cvxpy.T @ Cov @ w_cvxpy
    # print("CVXPY (MOSEK): ", mosek)

    # ILS Enhanced VQE
    ils = ILSSolver(Cov, args.sampler)
    ansatz = TwoLocal(num_qubits=ils.vqe.num_qubits, 
                                    rotation_blocks=['ry','rz'], 
                                    entanglement_blocks='cz',
                                    reps=args.rep,
                                    entanglement='full')
    ils.compute_seq(args.N, ansatz.num_parameters_settable)

    opt = COBYLA(maxiter=args.maxiter, tol=0.1)

    # Prepare the quantum instance
    if args.backend == "GPU":
        backend = Aer.get_backend(args.backend_name)
        try:
            backend.set_options(device='GPU')
        except AerError as e:
            print(e)
    elif args.backend == "IBMQ":
        from qiskit import IBMQ
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=args.hub, group=args.group, project=args.project)
        backend = provider.get_backend(args.backend_name)
    else:
        backend = Aer.get_backend(args.backend_name)
    qi = QuantumInstance(backend, seed_transpiler=args.seed, seed_simulator=args.seed, shots=args.shots)     

    ils_data = ils.solve(ansatz= ansatz,
                opt= opt,
                qi= qi
            )
    ils_Err = [np.abs(1 - np.min(data["values"])/mosek) for data in ils_data]

    # Plot Cost function
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for n in range(ils.samples.shape[0]):
        ax1.plot(ils_data[n]['counts'], ils_data[n]['values'])
    ax1.hlines(mosek, ils_data[0]['counts'][0], ils_data[0]['counts'][-1], label="MOSEK Optimum", color="black")
    ax1.set_xlabel('Eval count')
    ax1.set_ylabel('Value')

    # Boxplot Error
    err_data = [100*np.array(err) for err in ils_Err]
    print("The minimum relative error: {}%.".format(np.min(err_data)))
    ax2.boxplot(err_data)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Relative Error (%)')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    # Benchmark parameters
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--N", type=int, default=128)

    # Quantum Solver parameters
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--sampler", type=str, default="sobol", choices=["sobol", "random"])

    # Quantum Instance
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--backend_name", type=str, default="aer_simulator")
    parser.add_argument("--backend", type=str, default="simulator", choices=["GPU", "IBMQ", "simulator"])
    parser.add_argument("--hub", type=str, default='ibm-q')
    parser.add_argument("--group", type=str, default='open')
    parser.add_argument("--project", type=str, default='main')
    parser.add_argument("--job_name", type=str, default='benchmark')

    args = parser.parse_args()
    main(args)