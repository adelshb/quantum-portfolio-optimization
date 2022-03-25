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

""" CVXPY Optimization Method."""

from typing import Optional
import cvxpy as cp
import numpy as np
from numpy import ndarray

def CVXPYSolver(Cov:ndarray,
                mu:ndarray,
                gamma:float = 0.1,
                budget: float = 1000,
                asset_limit: float = 1.0,
                verbose: Optional[bool] = False
                ) -> float:
    """
    Take a Covariance matrix (from the different assets) and minimize the risk via Quadratic Programming.

    Args:
        Cov : Covariance matrix
        mu : Assets' forecasts returns
        gamma : Risk aversion coefficient
        budget : Maximum budget
        asset_limit : Maximum fraction of budget allocation per asset (1 = no limit)
    Returns:
        w : Optimum solution. w[i] is the asset allocation for asset i.
    """

    # Define and solve the CVXPY problem.
    w = cp.Variable(Cov.shape[0], nonneg=True)
    
    constraints = [cp.sum(w) <= budget]
    constraints += [np.eye(Cov.shape[0]) @ w <= asset_limit * budget * np.ones(Cov.shape[0])]

    prob = cp.Problem(cp.Minimize(gamma * 0.5 * cp.quad_form(w,Cov) - mu.T @ w),
                    constraints)

    prob.solve(solver=cp.MOSEK, verbose=verbose)
    return prob.value, w.value