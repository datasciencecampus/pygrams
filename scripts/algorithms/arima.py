import warnings

import numpy as np
from numpy import clip, inf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA


class ARIMAForecast(object):

    def __evaluate_models(self, dataset, p_values, d_values, q_values):
        dataset=np.array(dataset)
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        mse = self.__evaluate_arima_model(dataset, order, ground_truth_in_history=True)
                        if mse < best_score:
                            best_score = mse
                            best_cfg = order
                    except:
                        continue
        return best_cfg, best_score

    def __evaluate_arima_model(self, X, arima_order, ground_truth_in_history=False):

        train_ratio = 0.8
        train_size = int(len(X) * train_ratio)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0, maxiter=200)
            yhat = model_fit.forecast()[0][0]
            predictions.append(yhat)
            history.append(test[t] if ground_truth_in_history else yhat)
        error = mean_squared_error(test, predictions)
        return error

    def __arima_model_predict(self, X, arima_order, steps_ahead):
        # make predictions
        predictions = list()
        try:
            for t in range(steps_ahead):
                model = ARIMA(X, order=arima_order)
                model_fit = model.fit(disp=0)
                yhat = model_fit.forecast()[0][0]
                predictions.append(yhat)
                X = np.append(X, yhat)
        except:
            predictions.extend([np.nan] * (steps_ahead - len(predictions)))

        return predictions

    def __init__(self, data_in, num_prediction_periods ):
        if not all(isinstance(x, float) for x in data_in):
            raise ValueError('Time series must be all float values')

        p_values = [0, 1, 2, 4, 6]
        d_values = range(0, 3)
        q_values = range(0, 3)
        warnings.filterwarnings("ignore")
        self.__order, score = self.__evaluate_models(data_in, p_values, d_values, q_values)
        self.__predictions = self.__arima_model_predict(data_in, self.__order, num_prediction_periods)

    @property
    def configuration(self):
        return self.__order

    def predict_counts(self):
        return clip(self.__predictions, 0, inf)
