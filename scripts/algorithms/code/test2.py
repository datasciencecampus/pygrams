import pandas as pd
import numpy as np
from scripts.algorithms.code.ssm import SteadyStateModel
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.interactive(False)

# data = pd.read_csv('/Users/michaelhodge/GitProjects/pyGrams/scripts/algorithms/code/emergent_time_series.csv', header=None)
# data = pd.read_csv('/Users/michaelhodge/GitProjects/pyGrams/scripts/algorithms/code/declining_time_series.csv', header=None)
timeseries = [2.0, 8.0, 4.0, 7.0, 4.0, 10.0, 6.0, 10.0,
       8.0, 11.0, 4.0, 5.0, 5.0, 13.0, 6.0, 6.0, 9.0, 6.0, 12.0, 15.0,
       9.0, 15.0, 15.0, 12.0, 18.0, 17.0, 18.0, 15.0, 18.0, 18.0, 14.0,
       9.0, 11.0, 8.0, 14.0, 26.0, 24.0, 18.0, 18.0, 17.0, 8.0, 18.0,
       17.0, 17.0, 12.0, 16.0, 16.0, 30.0, 16.0, 24.0, 29.0, 25.0, 16.0,
       19.0]

# timeseries = np.array([323,340,296,217,265,337,326,284,276,294,252,253,
#                264,190,262,256,264,232,211,222,235,219,273,292,
#                330,268,260,263,277,309,282,316,348,314,314,317,
#                350,368,375,321,413,395,368,330,407,316,349,377,
#                320,334,340,317])
sigma_gnu = [0.01,0.1,0.5,1]
sigma_eta = [0.01,0.1,0.5,1]
delta = [0.5,0.7,0.9]


ssm = SteadyStateModel(timeseries)
opt_param, dfk_out, alphahat, mse_alphahat = ssm.run_smoothing(sigma_gnu, sigma_eta, delta)
smoothed_data = np.squeeze(np.asarray(alphahat[0,:]))

print(smoothed_data)
