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

from solver.cvxpy.cvxpy_solver import CVXPYSolver
from data_factory.utils import rand_data
from data_factory.market import Market

__available_methods = ["random", "brownian_motion", "loading"]

def main(args):

    if args.data_method == "brownian_motion" or args.data_method == "random":
        data = rand_data(args.num_assets , args.time_period, normalize_data=args.normalize_data, method=args.data_method)
    elif args.data_type == "loading":
        with open(args.path, 'rb') as f:
            data = np.load(f)

    market = Market(data, normalize_data=args.normalize_data)

    # Portfolio optimization parameters
    Cov = market.Cov
    mu = market.mu

    # CVXPY
    gammas = np.logspace(-5, 2, num=10)
    risk = []
    ret = []
    wmax = []
    for gamma in gammas:
        w_cvxpy = CVXPYSolver(Cov=Cov, mu=mu, gamma=gamma, verbose = False)
        try :
            risk.append(w_cvxpy.T @ Cov  @ w_cvxpy)
            ret.append(mu.T @ w_cvxpy)
            wmax.append(w_cvxpy.max())
        except:
            print("CVXPY failed for {}".format(gamma))
            risk.append(np.nan)
            ret.append(np.nan)
            wmax.append(np.nan)

    # Plot simulated price paths
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    array_day_plot = [t for t in range(args.time_period)]
    for n in range(args.num_assets):
        ax.plot(array_day_plot, market.X[n], label="Asset {}".format(n))
    plt.grid()
    plt.xlabel('Day')
    plt.ylabel('Asset price')
    plt.legend(loc='best')
    plt.show()

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gammas, risk, label='Risk')  # Plot some data on the axes.
    ax.plot(gammas, ret, label='Return')  # Plot more data on the axes...
    ax.plot(gammas, wmax, label='Max weight')  # Plot more data on the axes...
    ax.set_xlabel('Risk aversion (gamma)')  # Add an x-label to the axes.
    ax.legend();  # Add a legend.
    ax.set_title("MOSEK with num assets = {}".format(args.num_assets))
    ax.set_yscale('log')
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--num_assets", type=int, default=2)
    parser.add_argument("--time_period", type=int, default=256)
    parser.add_argument("--data_method", type=str, default="brownian_motion", choices=__available_methods)
    parser.add_argument("--normalize_data", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="datasets/data.csv")

    # Portfolio Optimization parameters
    parser.add_argument("--gamma", type=float, default=0.1)

    args = parser.parse_args()
    main(args)