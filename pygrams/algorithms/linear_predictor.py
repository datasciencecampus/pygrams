import numpy as np
from numpy import clip, inf
from sklearn import linear_model


class LinearPredictor(object):

    def __init__(self, data_in, num_prediction_periods):
        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

        self.__model = linear_model.LinearRegression()
        X = np.array(range(len(data_in))).reshape(-1, 1)
        y = np.array(data_in).reshape(-1, 1)
        self.__model.fit(X, y)

    @property
    def configuration(self):
        return ""

    def predict_counts(self):
        start_index = len(self.__history)
        X = np.array(range(start_index, start_index + self.__num_prediction_periods)).reshape(-1, 1)
        y = self.__model.predict(X)
        y_list = y.ravel().tolist()
        return clip(y_list, 0, inf)
