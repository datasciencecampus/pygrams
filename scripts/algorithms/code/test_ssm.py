import unittest
from scripts.algorithms.code.ssm import SteadyStateModel
import numpy as np

class TestSSM(unittest.TestCase):

    def setUp(self):
        self.__timeseries = [2.0, 8.0, 4.0, 7.0, 4.0, 10.0, 6.0, 10.0,
       8.0, 11.0, 4.0, 5.0, 5.0, 13.0, 6.0, 6.0, 9.0, 6.0, 12.0, 15.0,
       9.0, 15.0, 15.0, 12.0, 18.0, 17.0, 18.0, 15.0, 18.0, 18.0, 14.0,
       9.0, 11.0, 8.0, 14.0, 26.0, 24.0, 18.0, 18.0, 17.0, 8.0, 18.0,
       17.0, 17.0, 12.0, 16.0, 16.0, 30.0, 16.0, 24.0, 29.0, 25.0, 16.0,
       19.0]
        self.__sigma_gnu = [0.01, 0.1, 0.5, 1]
        self.__sigma_eta = [0.01, 0.1, 0.5, 1]
        self.__delta = [0.5, 0.7, 0.9]
        self.__timeseries_smoothed = [1.7889245, 7.54870676, 6.08985986, 6.68350498, 6.61723599, 7.3570212,
                                      7.30761325, 7.79724598, 7.70169983, 7.84197659, 7.16922453, 7.24786526,
                                      7.45663062, 8.3514342, 8.09252576, 8.39533882, 9.08798232, 9.55235424,
                                      10.7965019, 11.80141528, 12.06592993, 13.17324884, 13.8073726, 14.18358281,
                                      15.17143447, 15.4985337, 15.77638624, 15.57513721, 15.67282105, 15.35006534,
                                      14.60336766, 13.87328206, 14.04035223, 14.37279729, 15.8029767, 17.66585622,
                                      18.01753704, 17.51512684, 17.12086369, 16.62077081, 15.71245519, 16.51029556,
                                      16.6383978, 16.89152218, 16.88682743, 17.88779656, 18.82330016, 20.79400621,
                                      20.4765224, 21.61104188, 22.20536325, 21.73745322, 20.65603475, 20.5764788]

        self.__timeseries_smoothed = [round(e, 3) for e in self.__timeseries_smoothed]

    def test_ssm(self):
        ssm = SteadyStateModel(self.__timeseries)
        opt_param, dfk_out, alphahat, mse_alphahat = ssm.run_smoothing(self.__sigma_gnu, self.__sigma_eta, self.__delta)
        smoothed_data = np.squeeze(np.asarray(alphahat[0, :]))
        smoothed_data = np.round(smoothed_data, 3)
        self.assertEquals(smoothed_data.tolist(), self.__timeseries_smoothed)


if __name__ == '__main__':
    unittest.main()