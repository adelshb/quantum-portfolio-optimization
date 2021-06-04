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

def GekkoSolver(H):

    # initialize Model
    m = GEKKO()

    # initialize variable
    w = m.Array(m.Var,(H.shape[0]))

    # intial guess
    #ig = [1,5,5,1]

    # upper and lower bounds
    i = 0
    for wi in w:
        #xi.value = ig[i]
        wi.lower = 0
        wi.upper = 1

    # Equations
    m.Equation(sum(w)==1)

    #Objective
    m.Obj(w.T @ H @ w)
    m.solve()
    print(w)

    return w.value