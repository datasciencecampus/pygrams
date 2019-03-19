from numpy import clip, inf


class NaivePredictor(object):

    def __init__(self, data_in, num_prediction_periods):
        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

    @property
    def configuration(self):
        return ""

    def predict_counts(self):
        y_list = self.__history[-1:] * self.__num_prediction_periods
        return clip(y_list, 0, inf)
