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

def main(args):

    Cov = randcovmat(args.d)
    Cov = np.array([[1,0],[0,0]])

    # CVXPY
    w_cvxpy = CVXPYSolver(Cov)
    print("CVXPY: ", w_cvxpy.T @ Cov @ w_cvxpy)

    # VQE
    res_vqe = VQESolver(Cov = Cov, Nq = args.Nq)
    print(res_vqe)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--Nq", type=int, default=3)

    args = parser.parse_args()
    main(args)