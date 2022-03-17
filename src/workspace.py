from data_factory.market import Market
from data_factory.utils import rand_data, randcovmat

N = 2
T = 36
data = rand_data(N, T, method="brownian_motion")

from IPython import embed; embed()
market = Market(data)

Cov = randcovmat(N)

from IPython import embed; embed()