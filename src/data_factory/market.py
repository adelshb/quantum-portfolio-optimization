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

from typing import Optional, List, Union, Callable

import numpy as np
from numpy import ndarray

from pandas import DataFrame

from data_factory.utils import mean_forcast_return

class Market():
    "Class to handle data and compute model parameters from data."

    def __init__(self,
        data: Union[ndarray, DataFrame],
        forcast_return: Callable = mean_forcast_return,
        ) -> None:

        # Get parameters
        if isinstance(data, ndarray):
            self.df = None
            self.X = data
            self.N = data.shape[0]
            self.T = data.shape[1]
        elif isinstance(data, DataFrame):
            self.df = data
            self.X = data.values.T
            self.N = data.shape[1]
            self.T = data.shape[0]

        self.Cov = self.cov() # Total covariance matrix
        self.mu = self.forcast_return(forcast_return= forcast_return) # Total forcast return

    def cov(self,
        assets: Optional[List] = None,
        Ti: Optional[Union[int,str]] = 0,
        Tf: Optional[Union[int,str]] = -1,
        ) -> ndarray:
        """Compute covariance matrix for a specific time period and for a list of specified assets.
        Args:
            assets: List of assets to compute covariance matrix for.
            Ti: Initial time step
            Tf: Final time step.
        Returns:
            Cov: Covariance matrix.
        """

        # If no assets is specified, compute covariance matrix for all assets
        if assets is None:
            assets = [i for i in range(self.N)]

        # Compute covariance matrix
        if isinstance(Ti, str):
            try:
                Ti = self.df.index.get_loc(Ti)
            except:
                raise ValueError("Ti is not a valid date or not Dataframe with dates was provided.")
        if isinstance(Tf, str):
            try:
                Tf = self.df.index.get_loc(Tf)
            except:
                raise ValueError("Tf is not a valid date or not Dataframe with dates was provided.")

        Cov = np.cov(self.X[assets, Ti:Tf])
        return Cov

    def forcast_return(self,
        forcast_return: Callable = mean_forcast_return,
        assets: Optional[List] = None,
        Ti: Optional[Union[int,str]] = 0,
        Tf: Optional[Union[int,str]] = -1,
        
        ) -> ndarray:
        """Compute assets' forcast return for a specific time period and for a list of specified assets.
        Args:
            assets: List of assets to compute covariance matrix for.
            Ti: Initial time step.
            Tf: Final time step.
        Returns:
            forcast_return: Forcast return for specifed assets.
        """

        # If no assets is specified, compute covariance matrix for all assets
        if assets is None:
            assets = [i for i in range(self.N)]

        # Compute forcast return
        if isinstance(Ti, str):
            try:
                Ti = self.df.index.get_loc(Ti)
            except:
                raise ValueError("Ti is not a valid date or not Dataframe with dates was provided.")
        if isinstance(Tf, str):
            try:
                Tf = self.df.index.get_loc(Tf)
            except:
                raise ValueError("Tf is not a valid date or not Dataframe with dates was provided.")

        fr = forcast_return(self.X[assets, Ti:Tf]) 
        return fr