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

from typing import Optional

import numpy as np
from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from data_factory.brownian import Brownian

def randcovmat(N: int, seed: Optional[int]= None)-> ndarray:
    """
    Generate a random covariant matrix
    Args:
        N: number of assets
        seed: seed for random number generator
    Returns:
        A: random covariant matrix
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    A = np.random.rand(N, N)
    return A @ A.T

def rand_data(N: int,
    T: int,
    seed: Optional[int] = None,
    freq: Optional[int] = "MS",
    init_date: Optional[str] = '2019-01-01',
    method: str = "random",
    normalize_data: bool = True,
    ) -> DataFrame:
    """
    Generate random time series with dates
    Args:
        N: number of assets
        T: number of time steps
        seed: seed for random number generator
        feq: frequency of dates
        init_date: initial date
        method: method to generate random data
    Returns:
        X: Dataframe with random time series and dates
    """

    __available_methods = ["random", "brownian_motion"]
    if method not in __available_methods:
        raise ValueError(f"method should be one of {__available_methods}")

    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    rng = pd.date_range(init_date, freq=freq, periods=T)

    if method == "random":
        X = np.random.rand(T, N)
    elif method == "brownian_motion":
        X = []
        for __ in range(N):
            s0 = np.random.uniform(low=50, high=400, size=(1,))
            X.append(Brownian().stock_prices(s0=s0, n_step=T))
        X = np.array(X).T
    
    if normalize_data == True:
        X = (X - X.min())/(X.max()-X.min())

    df = pd.DataFrame(X, 
                  columns=["Asset_" + str(i) for i in range(N)], 
                  index=rng)
    return df

