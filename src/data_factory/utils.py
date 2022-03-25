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
    freq: Optional[int] = "D",
    init_date: Optional[str] = '2021-01-01',
    method: str = "random",
    low: Optional[int] = 50,
    high: Optional[int] = 400,
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
        low: lower bound initial sotck prices
        high: upper bound initial sotck prices
    Returns:
        Dataframe with random time series and dates
    """

    __available_methods = ["random", "brownian_motion"]

    if method not in __available_methods:
        raise ValueError(f"method should be one of {__available_methods}")

    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    rng = pd.date_range(init_date, freq=freq, periods=T)

    if method == "random":
        X = np.random.rand(N, T)

    elif method == "brownian_motion":
        br = Brownian(N)
        X = br.generate_prices(T= T,
                r= 0.001,
                dt= 1.0/T,
                low= low,
                high= high)

    df = pd.DataFrame(X.T, 
                  columns=["Asset_" + str(i) for i in range(N)], 
                  index=rng)

    return df

def mean_forcast_return(X: ndarray) -> ndarray:

    forcast_return = X[:,-1] / np.mean(X, axis=1)
    return forcast_return