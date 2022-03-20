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

from scipy.stats import random_correlation

class Brownian():
    """
    A Class for Brownian motion for simulated multi-asset baskets with correlated prices
    """
    def __init__(self,
        num_assets: int = 10,
        correlations: Optional[ndarray] = None,
        init_prices: Optional[ndarray] = None,
        volatilities: Optional[ndarray] = None,
        ) -> None:
        """
        Init class
        Args:
            num_assets: Number of assets
            correlations: Correlation matrix
            init_prices: Initial prices
            volatilities: Volatilities
        """
        
        # Get parameters
        self.num_assets = num_assets

        self.Corr = correlations
        if isinstance(self.Corr, ndarray):
            self.R = np.linalg.cholesky(self.Corr)
        else:
            self.R = None

        self.init_prices = init_prices
        self.volatilities = volatilities

    
    def rand_corr(self) -> None:
        """
        Generate a random correlation matrix and its cholesky decomposition
        """

        # Generate a random correlation matrix from random eigenvalues
        rng = np.random.default_rng()
        tmp_eigs = np.abs(np.random.rand(self.num_assets))
        eigs = self.num_assets * tmp_eigs / np.sum(tmp_eigs)

        self.Corr = random_correlation.rvs(eigs, random_state=rng)

        # Perform Cholesky decomposition on correlation matrix
        self.R = np.linalg.cholesky(self.Corr)

    def generate_prices(self,
        T: int = 256,
        r: float = 0.001,
        dt: float = 0.004,
        low: Optional[int] = 100,
        high: Optional[int] = 400
        ) -> ndarray:
        """
        Generate prices for a Brownian motion
        Args:
            T: Number of simulated days
            r : Risk free rate (annual)
            dt: Time increment (annualized)
            low: Lowest price
            high: Highest price
        """

        # Test if initial parameters are given. Otherwise generate random parameters
        if isinstance(self.init_prices, ndarray):
            stock_prices = np.array([[s]*T for s in self.init_prices])
        else:
            init_prices = np.random.randint(low=low, high=high, size=(self.num_assets, 1))
            stock_prices = np.array([[s]*T for s in init_prices])

        if self.volatilities is None:
            mu, sigma = 0.5, 0.05
            self.volatilities = np.random.normal(mu, sigma, self.num_assets)

        if self.R is None:
            self.rand_corr()

        for t in range(1, T):
            # Generate array of random standard normal draws
            random_array = np.random.standard_normal(self.num_assets)
            
            # Multiply R with random_array to obtain correlated epsilons
            epsilon_array = np.inner(random_array, self.R)

            # Sample price path per stock
            for n in range(self.num_assets):
                dt = 1 / T 
                S = stock_prices[n,t-1]
                v = self.volatilities[n]
                epsilon = epsilon_array[n]
                
                # Generate new stock price
                stock_prices[n,t] = S * np.exp((r - 0.5 * v**2) * dt + v * np.sqrt(dt) * epsilon)

        return stock_prices.reshape(self.num_assets, T)