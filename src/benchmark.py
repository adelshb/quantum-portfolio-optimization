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
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.providers.aer import AerError
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import COBYLA

from solver.cvxpy.cvxpy_solver import CVXPYSolver
from solver.ils.ils import ILSSolver
from data_factory.utils import rand_data
from data_factory.market import Market

__available_methods = ["random", "brownian_motion", "loading"]

def main(args):

    if args.data_method == "brownian_motion" or args.data_method == "random":
        data = rand_data(args.num_assets , args.time_period , method=args.data_method)
    elif args.data_type == "loading":
        with open(args.path, 'rb') as f:
            data = np.load(f)

    market = Market(data)

    # Portfolio optimization parameters
    Cov = market.Cov
    mu = market.mu
    # mu = np.zeros((args.num_assets)) # Minimize the risk

    cvxpy, __ = CVXPYSolver(Cov=Cov, mu=mu, gamma=args.gamma, verbose=False)
    print("CVXPY: {}".format(cvxpy))

    # Prepare quantum instance for benchmark
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

    # ILS Enhanced VQE
    ils = ILSSolver(Cov=Cov, 
                mu=mu, 
                gamma=args.gamma,
                budget=args.budget,
                asset_limit=args.asset_limit,
                sampler_method=args.sampler)

    ansatz = TwoLocal(num_qubits=ils.vqe.num_qubits, 
                                    rotation_blocks=['ry','rz'], 
                                    entanglement_blocks='cz',
                                    reps=args.rep,
                                    entanglement='full')

    ils.compute_seq(args.N, ansatz.num_parameters_settable)

    opt = COBYLA(maxiter=args.maxiter, tol=0.1)  

    ils_data = ils.solve(ansatz= ansatz,
                opt= opt,
                qi= qi)

    print("ILS (COBYLA): ", min([np.min(data["values"]) for data in ils_data]))
    ils_Err = [np.abs(1 - np.min(data["values"])/cvxpy) for data in ils_data]

    # Plot Cost function
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for n in range(ils.samples.shape[0]):
        ax1.plot(ils_data[n]['counts'], ils_data[n]['values'])
    ax1.hlines(cvxpy, ils_data[0]['counts'][0], ils_data[0]['counts'][-1], label="MOSEK Optimum", color="black")
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

    # Dataset
    parser.add_argument("--num_assets", type=int, default=2)
    parser.add_argument("--time_period", type=int, default=256)
    parser.add_argument("--data_method", type=str, default="brownian_motion", choices=["brownian_motion", "random", "load"])
    parser.add_argument("--data_path", type=str, default="datasets/data.csv")

    # Portfolio Optimization parameters
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--budget", type=float, default=100)
    parser.add_argument("--asset_limit", type=float, default=1.0)

    # Benchmark parameters
    parser.add_argument("--N", type=int, default=8)

    # Quantum Solver parameters
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--sampler", type=str, default="random", choices=["sobol", "random"])

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