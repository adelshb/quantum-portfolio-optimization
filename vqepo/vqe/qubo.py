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
    Recast the optimization as a Quadratic Unconstrained Binary Optimization (QUBO) via binary encoding of each variable.
"""

import numpy as np 
def bin_enc(Nq, Cov):
    """
    Input
    ----------
    Nq : Each variable is encoded on Nq bits.
    Cov : Covariance matrix

    Output
    ----------
    Q : QUBO matrix
    """

    K = 2**Nq - 1
    #B = np.tile([2**n for n in range(Nq)], (Nq, 1))
    B = np.array([2**n for n in range(Nq**2)]).reshape((Nq,Nq))

    Q = np.kron(Cov, B)/(K**2)
    return Q