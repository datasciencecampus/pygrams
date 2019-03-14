import os

import pandas as pd
from tqdm import tqdm

from scripts.algorithms.emergence import Emergence
from scripts.utils.Utils_emtech import get_row_indices_and_values, timeseries_weekly_to_quarterly
from scripts.vandv.emergence_labels import map_prediction_to_emergence_label, report_predicted_emergence_labels_html
from scripts.vandv.graphs import report_prediction_as_graphs_html
from scripts.vandv.predictor import evaluate_prediction


# 'USPTO-granted-full-random-500000-term_counts.pkl.bz2'
class Pipeline_emtech(object):
    def __init__(self, term_counts_filename, m_steps_ahead=5, curves=True, nterms=50, minimum_patents_per_quarter=20):
        self.__M = m_steps_ahead

        [self.__term_counts_per_week, self.__term_ngrams, self.__number_of_patents_per_week,
         self.__weekly_iso_dates] = pd.read_pickle(
            os.path.join(term_counts_filename))
        self.__output_folder = os.path.join('outputs', 'emtech')
        self.__base_file_name = os.path.basename(term_counts_filename)

        term_counts_per_week_csc = self.__term_counts_per_week.tocsc()

        em = Emergence(self.__number_of_patents_per_week)
        self.__emergence_list = []
        for term_index in tqdm(range(self.__term_counts_per_week.shape[1]), unit='term', desc='Calculating eScore',
                               leave=False, unit_scale=True):
            term_ngram = self.__term_ngrams[term_index]
            row_indices, row_values = get_row_indices_and_values(term_counts_per_week_csc, term_index)

            if len(row_values) == 0:
                continue

            weekly_iso_dates = [self.__weekly_iso_dates[x] for x in row_indices]

            _, quarterly_values = timeseries_weekly_to_quarterly(weekly_iso_dates, row_values)
            if max(quarterly_values) < minimum_patents_per_quarter:
                continue

            if em.init_vars(row_indices, row_values):
                escore = em.calculate_escore() if not curves else em.escore2()
                self.__emergence_list.append((term_ngram, escore))

        self.__emergence_list.sort(key=lambda emergence: -emergence[1])
        # for tup in self.__emergence_list:
        #     print(tup[0] + ": " + str(tup[1]))

        self.__emergent = [x[0] for x in self.__emergence_list[:nterms]]
        self.__declining = [x[0] for x in self.__emergence_list[-nterms:]]

        zero_pivot_emergence = None
        last_emergence = self.__emergence_list[0][1]
        for index, value in enumerate(self.__emergence_list[1:]):
            if value[1] <= 0.0 < last_emergence:
                zero_pivot_emergence = index
                break
            last_emergence = value[1]

        stationary_start_index = zero_pivot_emergence - nterms // 2
        stationary_end_index = zero_pivot_emergence + nterms // 2
        self.__stationary = [x[0] for x in self.__emergence_list[stationary_start_index:stationary_end_index]]

        print()
        print('Emergent')
        for tup in self.__emergence_list[:nterms]:
            print(tup[0] + ": " + str(tup[1]))
        print()

        print('Stationary')
        for tup in self.__emergence_list[stationary_start_index:stationary_end_index]:
            print(tup[0] + ": " + str(tup[1]))
        print()

        print('Declining')
        for tup in self.__emergence_list[-nterms:]:
            print(tup[0] + ": " + str(tup[1]))
        print()

        # construct a terms list for n emergent n stationary? n declining

    def run(self, predictors_to_run, emergence, normalized=False, train_test=False):
        if emergence == 'emergent':
            terms = self.__emergent
        elif emergence == 'stationary':
            terms = self.__stationary
        elif emergence == 'declining':
            terms = self.__declining
        else:
            raise ValueError(f'Unrecognised value for emergence_type: {emergence}')

        html_results = ''

        results, training_values, test_values = evaluate_prediction(self.__term_counts_per_week, self.__term_ngrams,
                                                                    predictors_to_run,
                                                                    self.__weekly_iso_dates,
                                                                    self.__output_folder, test_terms=terms,
                                                                    prefix=self.__base_file_name,
                                                                    suffix=emergence,
                                                                    number_of_patents_per_week=self.__number_of_patents_per_week,
                                                                    num_prediction_periods=self.__M,
                                                                    normalised=normalized,
                                                                    test_forecasts=train_test)

        predicted_emergence = map_prediction_to_emergence_label(results, training_values, test_values, predictors_to_run, test_terms=terms)

        html_results += report_predicted_emergence_labels_html(predicted_emergence)

        html_results += report_prediction_as_graphs_html(results, predictors_to_run, self.__weekly_iso_dates,
                                                         test_values=test_values,
                                                         test_terms=terms, training_values=training_values,
                                                         normalised=normalized,
                                                         test_forecasts=train_test)

        return html_results
