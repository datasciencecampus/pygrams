# from scripts.algorithms.state_space import StateSpaceModel
import numpy as np

from scripts.algorithms.arima import ARIMAForecast
from scripts.algorithms.holtwinters_predictor import HoltWintersPredictor
from scripts.algorithms.linear_predictor import LinearPredictor
from scripts.algorithms.lstm import LstmForecasterMultipleLookAhead, LstmForecasterSingleLookAhead, \
    LstmForecasterMultipleModelSingleLookAhead
from scripts.algorithms.naive_predictor import NaivePredictor
from scripts.algorithms.polynomial_predictor import PolynomialPredictor


class PredictorFactory(object):

    @staticmethod
    def predictor_factory(predictor_name, data_name, training_values, num_prediction_periods):
        if predictor_name == 'ARIMA':
            return ARIMAForecast(training_values, num_prediction_periods)
        elif predictor_name == 'LSTM-multiLA-stateful':
            return LstmForecasterMultipleLookAhead(np.array(training_values), num_prediction_periods,
                                                   num_prediction_periods,
                                                   data_name, True, 42)
        elif predictor_name == 'LSTM-multiLA-stateless':
            return LstmForecasterMultipleLookAhead(np.array(training_values), num_prediction_periods,
                                                   num_prediction_periods,
                                                   data_name, False, 42)
        elif predictor_name == 'LSTM-1LA-stateful':
            return LstmForecasterSingleLookAhead(np.array(training_values), num_prediction_periods,
                                                 num_prediction_periods,
                                                 data_name, True, 42)
        elif predictor_name == 'LSTM-1LA-stateless':
            return LstmForecasterSingleLookAhead(np.array(training_values), num_prediction_periods,
                                                 num_prediction_periods,
                                                 data_name, False, 42)
        elif predictor_name == 'LSTM-multiM-1LA-stateful':
            return LstmForecasterMultipleModelSingleLookAhead(np.array(training_values), num_prediction_periods,
                                                              num_prediction_periods,
                                                              data_name, True, 42)
        elif predictor_name == 'LSTM-multiM-1LA-stateless':
            return LstmForecasterMultipleModelSingleLookAhead(np.array(training_values), num_prediction_periods,
                                                              num_prediction_periods,
                                                              data_name, False, 42)
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
        # elif predictor_name == 'state-space':
        #     return StateSpaceModel(training_values, num_prediction_periods)
        else:
            raise ValueError('Unknown predictor: ' + predictor_name)
