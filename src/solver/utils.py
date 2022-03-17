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
from numpy import ndarray

def new_inter(X: ndarray, a:float=0, b:float=1, c:float=0 , d:float=2*np.pi):

    X_new = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_new[i] = c + (X[i]-a) * (d-c) / (b-a) 
    return X_new