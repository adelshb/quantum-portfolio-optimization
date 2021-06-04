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
import numpy as np

from utils import randcovmat

from vqepo.gekko.gekko_solver import GekkoSolver
from vqepo.vqe.vqe_solver import VQESolver

def main(args):

    Cov = randcovmat(args.d)

    # True Value
    val = min(np.linalg.eig(Cov)[0])
    print(val)

    # Gekko
    res_gekko = GekkoSolver(Cov)
    print(res_gekko)

    # # VQE
    # res_vqe = VQESolver(Cov)
    # print(res_vqe)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--d", type=int, default=2)

    args = parser.parse_args()
    main(args)