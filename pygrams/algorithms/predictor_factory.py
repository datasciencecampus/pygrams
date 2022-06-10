# from scripts.algorithms.state_space import StateSpaceModel

from pygrams.algorithms.arima import ARIMAForecast
from pygrams.algorithms.holtwinters_predictor import HoltWintersPredictor
from pygrams.algorithms.linear_predictor import LinearPredictor
from pygrams.algorithms.naive_predictor import NaivePredictor
from pygrams.algorithms.polynomial_predictor import PolynomialPredictor
from pygrams.algorithms.state_space import StateSpaceModelObject


class PredictorFactory(object):

    @staticmethod
    def predictor_factory(predictor_name, data_name, training_values, num_prediction_periods):
        if predictor_name == 'ARIMA':
            return ARIMAForecast(training_values, num_prediction_periods)
        elif predictor_name == 'Naive':
            return NaivePredictor(training_values, num_prediction_periods)
        elif predictor_name == 'Linear':
            return LinearPredictor(training_values, num_prediction_periods)
        elif predictor_name == 'Quadratic':
            return PolynomialPredictor(training_values, num_prediction_periods)
        elif predictor_name == 'Cubic':
            return PolynomialPredictor(training_values, num_prediction_periods, degree=3)
        elif predictor_name == 'Holt-Winters':
            return HoltWintersPredictor(training_values, num_prediction_periods)
        elif predictor_name == 'SSM':
            return StateSpaceModelObject(training_values, num_prediction_periods)
        else:
            raise ValueError('Unknown predictor: ' + predictor_name)
