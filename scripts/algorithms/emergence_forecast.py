from scripts.algorithms.arima import ARIMAForecast


class EmergenceForecast(object):
    @staticmethod
    def factory(model_type):
        if model_type == "Arima":
            return ARIMAForecast()
        else:
            assert 0, "Bad model type: " + model_type
