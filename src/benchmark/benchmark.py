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

import numpy as np
from src.utils import randcovmat

from src.solver.gekko import gekko_solver
from src.solver.vqe import vqe_solver

d = 2
H = randcovmat(d)

val = min(np.linalg.eig(H)[0])

res_gekko = gekko_solver(H)
res_vqe = vqe_solver(H)
