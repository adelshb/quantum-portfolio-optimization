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

def bin_enc(Nq):
    """
    Input
    ----------
    Nq : Each variable is encoded on Nq bits.

    Output
    ----------
    Q : QUBO matrix
    """

    K = 2**Nq - 1

    return Q