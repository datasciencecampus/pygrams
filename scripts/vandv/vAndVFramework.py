import os

import pandas as pd
from tqdm import tqdm

from emerging_tech.scripts.algorithms.emergence import Emergence
from scripts.algorithms.emergence_forecast import EmergenceForecast


class VV(object):

    CUT_OFF_YEAR = 2011
    include_list = [241946,
                    4973710,
                    2711507,
                    815944,
                    915527,
                    396545,
                    4745680,
                    1129815,
                    1404874,
                    812656,
                    4783773,
                    747808,
                    4785581,
                    778199,
                    3443087,
                    3420193,
                    2438016,
                    812762,
                    2123699,
                    3425910,
                    751167,
                    4973887,
                    2411830,
                    1263111,
                    677209,
                    1389504,
                    1681016,
                    1259235,
                    3178531,
                    3944050,
                    4783217,
                    4041284,
                    1706852,
                    1785670,
                    4887407,
                    3968632,
                    2119849,
                    2123705,
                    4328194,
                    3178877,
                    1716630,
                    3484485,
                    4608696,
                    1707092,
                    3421621,
                    2118559,
                    2442104,
                    4331173,
                    1165903,
                    40435,
                    1057263,
                    2205683,
                    1184105,
                    3886207,
                    4329410,
                    1678604,
                    749469,
                    3055495,
                    1715942,
                    2207835,
                    3977569,
                    1161296,
                    4785195,
                    1722012,
                    907484,
                    3178944,
                    4471100,
                    812863,
                    2714165,
                    3982777,
                    2121569,
                    4975408,
                    4973761,
                    1804417,
                    2438050,
                    3968905,
                    1434533,
                    814882,
                    3922504,
                    1861810,
                    1069082,
                    3976937,
                    2438053,
                    4081087,
                    1905365,
                    1543116,
                    1434661,
                    915638,
                    1689710,
                    1727447,
                    1831704,
                    1693532,
                    78469,
                    4975875,
                    40443,
                    734641,
                    1676225,
                    4617431,
                    2559803,
                    3955866,
                    1918386,
                    3427038,
                    3314750,
                    1169137,
                    4040297,
                    4330793,
                    1395747,
                    2242098,
                    2713644,
                    70945,
                    817062,
                    905979,
                    921749,
                    4617170,
                    4212417,
                    1182936,
                    2411706,
                    914763,
                    2605369,
                    656158,
                    44149,
                    3403713,
                    2711251,
                    4974147,
                    2300913,
                    1183887,
                    4470906,
                    1186603,
                    2208963,
                    1610003,
                    2300920,
                    4209194,
                    813313,
                    3661284,
                    2986623,
                    1258922,
                    1691031,
                    485178,
                    3952341,
                    3178544,
                    1382310,
                    892029,
                    1672876,
                    3987727,
                    4617636,
                    364019,
                    2209672,
                    405419,
                    4471367,
                    3320003,
                    4594442,
                    2101101,
                    3230439,
                    1916910,
                    1389724,
                    2508603,
                    1190483,
                    4781749,
                    2853517,
                    1712079,
                    1865096,
                    4235953,
                    2395798,
                    3585201,
                    729565,
                    1723432,
                    1717461,
                    770561,
                    3550393,
                    1874744,
                    836110,
                    2121566,
                    1113552,
                    2635723,
                    3973384,
                    3028652,
                    3936606,
                    751804,
                    4524214,
                    3936593,
                    3984068,
                    3320614,
                    4957282,
                    3991819,
                    1918804,
                    2208966,
                    2395216,
                    1070760,
                    78688,
                    2850675,
                    1182702,
                    4090713,
                    2209678,
                    2632710,
                    2639055,
                    4863598,
                    3941794,
                    2206550,
                    1861237,
                    2711266,
                    43685,
                    2493433,
                    559781,
                    778958,
                    2839124,
                    3939648,
                    796842,
                    1715974,
                    812886,
                    4880832,
                    4205396,
                    4473786]

    def __init__(self, pickle_name, model_name):
        data_bundle = pd.read_pickle(os.path.join('data', pickle_name))
        self.__matrix = data_bundle[0]
        self.__feature_names = data_bundle[1]
        self.__week_dates = data_bundle[2]

        self.__model_name = model_name

        self.__expected = self.get_expected_dict()
        self.__predicted = self.get_prediction_dict(self.CUT_OFF_YEAR)

    def get_expected_dict(self):

        expected = {}
        matrix_csc = self.__matrix.tocsc()
        em = Emergence(self.__week_dates)

        for term_index in tqdm(range(0, len(matrix_csc.indptr) - 1), unit='term', leave=False):
            if term_index not in self.include_list:
                continue
            term = self.__feature_names[term_index]
            is_emergent, escore = em.calculate_escore(matrix_csc, term_index)

            start_idx_ptr = matrix_csc.indptr[term_index]
            end_idx_ptr = matrix_csc.indptr[term_index + 1]

            term_week_dates = []
            # iterate through non-zero indexes for dates, given a date for each row
            for ptr_index in range(start_idx_ptr, end_idx_ptr):
                row_index = matrix_csc.indices[ptr_index]
                term_week_dates.append(self.__week_dates[row_index])

            expected[term] = (term_week_dates, escore, is_emergent)
        return expected

    def get_prediction_dict(self, cut_off_year):

        predicted = {}
        matrix_csc = self.__matrix.tocsc()
        model = EmergenceForecast.factory(self.__model_name)
        for term_index in tqdm(range(0, len(matrix_csc.indptr) - 1), unit='term', leave=False):
            if term_index not in self.include_list:
                continue
            term = self.__feature_names[term_index]

            start_idx_ptr = matrix_csc.indptr[term_index]
            end_idx_ptr = matrix_csc.indptr[term_index + 1]

            term_week_dates = []
            # iterate through non-zero indexes for dates, given a date for each row
            for ptr_index in range(start_idx_ptr, end_idx_ptr):
                row_index = matrix_csc.indices[ptr_index]

                week_date = self.__week_dates[row_index]
                year = week_date / 100

                if year > cut_off_year:
                    break

                term_week_dates.append(week_date)

            nrecords_forecast = len((self.__expected[term])[0]) - len(term_week_dates)

            model.set_timeseries_weekly(term_week_dates)

            counts = model.predict_counts_weekly(nrecords_forecast)
            escore = model.predict_escore(term_week_dates)
            is_emergent = model.predict_emergence(term_week_dates)

            predicted[term] = (counts, escore, is_emergent, nrecords_forecast)
        return predicted


if __name__ == "__main__":
    v_v = VV('USPTO-random-500000-term_counts.pkl.bz2', 'Arima')
