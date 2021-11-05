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

from utils import new_inter, randcovmat
import numpy as np
from scipy.stats import qmc

from qpo.cvxpy.cvxpy_solver import CVXPYSolver

from qpo.vqe.vqe_solver import VQESolver

from qiskit import Aer
from qiskit.providers.aer import AerError

from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA#, L_BFGS_B, SLSQP, ADAM

import matplotlib.pyplot as plt

def main(args):

    Cov = randcovmat(args.d)

    # CVXPY
    w_cvxpy = CVXPYSolver(Cov, verbose = False)
    mosek = w_cvxpy.T @ Cov @ w_cvxpy
    # print("CVXPY (MOSEK): ", mosek)

    # Initialize VQE
    data = {}
    N = Cov.shape[0]
    vqe = VQESolver()
    vqe.qp(Cov = Cov)

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
    qi = QuantumInstance(backend, seed_transpiler=args.seed, seed_simulator=args.seed, shots=10000)
    
    optimizers = [COBYLA(maxiter=args.maxiter, tol=0.1)]

    vqe.to_ising()

    data = {}
    Err = {}
    for opt in optimizers:
        data[type(opt).__name__] = {}
        Err[type(opt).__name__]  = {}
        for rep in range(1,args.maxreps+1):
            data[type(opt).__name__][rep] = []
            Err[type(opt).__name__][rep] = []

            ansatz = TwoLocal(num_qubits=vqe.num_qubits, 
                                    rotation_blocks=['ry','rz'], 
                                    entanglement_blocks='cz',
                                    reps=rep,
                                    entanglement='full')
            if args.sampler == "sobol":                        
                samples = qmc.Sobol(d=ansatz.num_parameters_settable, scramble=False).random_base2(m=int(np.log2(args.N)))
            else:
                samples = np.random.uniform(low=0, high=1, size=(args.N, ansatz.num_parameters_settable))

            for samp in samples:

                init_weights = new_inter(samp)

                counts = []
                values = []
                param = []
                def store_intermediate_result(eval_count, parameters, mean, std):
                                    counts.append(eval_count)
                                    values.append(mean)
                                    param.append(parameters)

                vqe.vqe_instance(ansatz=ansatz,
                                optimizer=opt, 
                                init= init_weights,
                                quantum_instance=qi, 
                                callback=store_intermediate_result)

                vqe.solve()
                data[type(opt).__name__][rep].append({"reps": rep,
                                            "counts": np.asarray(counts),
                                            "values": np.asarray(values) + vqe.offset,
                                            "parameters": param})

                # print(np.min(values + vqe.offset))
                # Error according to MOSEK result
                Err[type(opt).__name__][rep].append(np.abs(mosek - np.min(values + vqe.offset))/mosek)

    # Plot Cost function
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for opt in optimizers:
        for rep in range(1,args.maxreps+1):
            # for n in range(args.N):
            for n in range(samples.shape[0]):
                label = type(opt).__name__ + "-reps-{}-{}".format(rep, n)
                ax1.plot(data[type(opt).__name__][rep][n]['counts'], data[type(opt).__name__][rep][n]['values'], label=label)
    ax1.hlines(mosek, data[type(optimizers[0]).__name__][1][0]['counts'][0], data[type(optimizers[0]).__name__][1][0]['counts'][-1], label="MOSEK Optimum", color="black")

    ax1.set_xlabel('Eval count')
    ax1.set_ylabel('Value')
    # plt.legend(loc='upper right')
    # plt.show()

    # Boxplot Error
    err_data = [100*np.array(Err[type(opt).__name__][rep]) for rep in range(1,args.maxreps+1)]
    print("The minimum relative error: {}%.".format(np.min(err_data)))
    # fig_box = plt.figure()
    for opt in optimizers:
        ax2.boxplot(err_data)

    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Relative Error (%)')
    # plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    # Benchmark parameters
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--N", type=int, default=128)

    # Quantum Solver parameters
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--maxreps", type=int, default=1)
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