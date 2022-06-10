import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class PolynomialPredictor(object):

    def __init__(self, data_in, num_prediction_periods, degree=2):
        self.__history = data_in
        self.__num_prediction_periods = num_prediction_periods

        polynomial_features = PolynomialFeatures(degree=degree,
                                                 include_bias=False)
        linear_regression = LinearRegression()
        self.__model = Pipeline([("polynomial_features", polynomial_features),
                                 ("linear_regression", linear_regression)])

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
        return np.clip(y_list, 0, np.inf)
