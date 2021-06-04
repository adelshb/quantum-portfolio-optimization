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

import cvxpy as cp  
import numpy as np

def CVXPYSolver(Cov):

    # Define and solve the CVXPY problem.
    w = cp.Variable(Cov.shape[0])
    constraints = [sum(w) == 1]
    for wi in w:
        constraints += [wi <= 1]
        constraints += [0 <= wi]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Cov)),
                    constraints)

    prob.solve(solver=cp.MOSEK, verbose=False)
    return w.value