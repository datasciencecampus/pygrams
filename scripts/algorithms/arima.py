from numpy import clip, inf
from pyramid.arima import auto_arima


class ARIMAForecast(object):

    def __init__(self, data_in, num_prediction_periods):
        if not all(isinstance(x, float) for x in data_in):
            raise ValueError('Time series must be all float values')

        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

        self.__stepwise_model = auto_arima(
            data_in,
            seasonal=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )

        self.__stepwise_model.fit(data_in)

    @property
    def configuration(self):
        return self.__stepwise_model.order

    def predict_counts(self):
        return clip(self.__stepwise_model.predict(n_periods=self.__num_prediction_periods), 0, inf)
