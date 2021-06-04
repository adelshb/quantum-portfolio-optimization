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
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli

def bin_enc(Nq, Cov):
    """
    Generate the QUBO matrix after binary encoding

    Args:
        Nq : Each variable is encoded on Nq bits.
        Cov : Covariance matrix

    Returns:
        Q : QUBO matrix
    """

    K = 2**Nq - 1
    B = np.array([2**n for n in range(Nq**2)]).reshape((Nq,Nq))
    Q = np.kron(Cov, B)/(K**2)

    return Q

def qubo2ising(Q):
    """
    Generate the Qubit operator of the Observable interpretable by the Qiskit VQE class
    
    Args:
        Q : QUBO matrix

    Returns
        H : Qubit operator of the Observable
    """
    N = Q.shape[0]
    H = 0
    for i in range(N):
        for j in range(N):
            pauli_str = ["I"] * N
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"
            pauli_str = "".join(pauli_str)
            H += PauliOp(Pauli(pauli_str), coeff= Q[i,j])
    return H