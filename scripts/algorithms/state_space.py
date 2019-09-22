import numpy as np

from scripts.algorithms.code.ssm import StateSpaceModel


class StateSpaceModelObject(object):

    def __init__(self, data_in, num_prediction_periods):
        if not all(isinstance(x, float) for x in data_in):
            raise ValueError('Time series must be all float values')

        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

        self.__alpha, self.__mse = StateSpaceModel(self.__history).run_smooth_forecast(k=self.__num_prediction_periods)
    @property
    def configuration(self):
        return None

    def predict_counts(self):
        return np.array(self.__alpha[0])[0]

    def predict_derivatives(self):
        return np.array(self.__alpha[1])[0]
