import numpy as np
from numpy import clip, inf
from statsmodels.tsa.holtwinters import Holt


class HoltWintersPredictor(object):

    def __init__(self, data_in, num_prediction_periods):
        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

        y = np.array(data_in).reshape(-1, 1)
        y = y + 0.000001  # HoltWinters doesn't like zeros

        self.__model = Holt(y, exponential=True, damped=True)
        self.__results = self.__model.fit(optimized=True)

    @property
    def configuration(self):
        return ""

    def predict_counts(self):
        start_index = len(self.__history)
        y = self.__model.predict(self.__results.params, start=start_index + 1, end=start_index + self.__num_prediction_periods)
        y_list = y.ravel().tolist()
        return clip(y_list, 0, inf)
