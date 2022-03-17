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

class Brownian():
    """
    A Class for Brownian motion using the Weiner process to build time series
    """
    def __init__(self) -> None:
        """
        Init class
        """
        pass
    
    def gen_normal(self,
        n_step: Optional[int] = 100
        ) -> ndarray:
        """
        Generate motion by drawing from the Normal distribution
        Args:
            n_step: Number of steps  
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.zeros(n_step)
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w

    def stock_prices(self,
        s0: float = 100,
        mu: float = 0.2,
        sigma: float = 0.68,
        n_step: int = 12,
        ) -> ndarray:
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
        Args:
            s0: Iniital stock price
            mu: 'Drift' of the stock (upwards or downwards)
            sigma: 'Volatility' of the stock
            n_step: Number of steps   
        Returns:
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        time_vector = np.linspace(0,1,num=n_step)
        # Stock variation
        stock_var = (mu-(sigma**2/2))*time_vector
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma*self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0*(np.exp(stock_var+weiner_process))

        return s