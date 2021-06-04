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

from gekko import GEKKO    
import numpy as np

def GekkoSolver(Cov):

    # Initialize model and variables
    m = GEKKO()
    w = m.Array(m.Var,(Cov.shape[0]))

    # Upper and lower bounds on variables
    for wi in w:
        wi.lower = 0
        wi.upper = 1

    # Sum of the weight is 1.
    m.Equation(sum(w)==1)

    # Objective 
    m.Obj(w.T @ Cov @ w)

    m.solve(disp=False)
    # print(w)
    # print(type(w))

    return w