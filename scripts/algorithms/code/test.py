import pandas as pd
import numpy as np
from scripts.algorithms.code.ssm import SteadyStateModel
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from statsmodels.nonparametric.smoothers_lowess import lowess

from tqdm import tqdm
plt.interactive(False)

# data = pd.read_csv('/Users/michaelhodge/GitProjects/pyGrams/scripts/algorithms/code/emergent_time_series.csv', header=None)
# data = pd.read_csv('/Users/michaelhodge/GitProjects/pyGrams/scripts/algorithms/code/declining_time_series.csv', header=None)
# data = pd.read_csv('/Users/michaelhodge/GitProjects/pyGrams/scripts/algorithms/code/stationary_time_series.csv', header=None)

def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

timeseries = np.array([323,340,296,217,265,337,326,284,276,294,252,253,
               264,190,262,256,264,232,211,222,235,219,273,292,
               330,268,260,263,277,309,282,316,348,314,314,317,
               350,368,375,321,413,395,368,330,407,316,349,377,
               320,334,340,317])

x = range(0,len(timeseries))

print(len(timeseries))
# sigma_gnu = [0.01,0.1,0.5,1]
# sigma_eta = [0.01,0.1,0.5,1]
# delta = [0.5,0.7,0.9]
#
# fig, axs = plt.subplots(5,5)
# fig.subplots_adjust(hspace = 1, wspace = .5)
# i = -1
# j = 0
# num = 0
# #
#
# # for index, row in tqdm(data.iterrows()):
# num = num + 1
# i = i + 1
# if i > 4:
#     j = j + 1
#     i = 0
# # timeseries = row.values[1:,]
# ssm = SteadyStateModel(timeseries)
# opt_param, dfk_out, alphahat, mse_alphahat = ssm.run_smoothing(sigma_gnu, sigma_eta, delta)
#
# if num == 1:
#     optimal_params = opt_param
# else:
#     optimal_params = np.vstack((optimal_params,opt_param))

# smoothed_data = np.squeeze(np.asarray(alphahat[0,:]))


window_size = round_up_to_odd(len(timeseries)/2)

polyn = 2

smoothed_data_2 = savgol_filter(timeseries, window_size, 3)
smoothed_data_3 = lowess(timeseries, x, window_size/len(timeseries), it=0)
plt.plot(timeseries)
plt.plot(smoothed_data_2)
plt.plot(smoothed_data_3[:,1])
# axs[i, j].title.set_text(row.values[0])

plt.show()