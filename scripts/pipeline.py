from os import path

import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import scripts.data_factory as data_factory
import scripts.output_factory as output_factory
import scripts.utils.date_utils
from scripts.algorithms.code.ssm import StateSpaceModel
from scripts.algorithms.emergence import Emergence
from scripts.algorithms.predictor_factory import PredictorFactory
from scripts.documents_filter import DocumentsFilter
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer, WordAnalyzer, lowercase_strip_accents_and_ownership
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import tfidf_subset_from_features, tfidf_from_text
from scripts.utils import utils
from scripts.vandv import ssm_reporting
from scripts.vandv.emergence_labels import map_prediction_to_emergence_label, report_predicted_emergence_labels_html
from scripts.vandv.predictor import evaluate_prediction
from scripts.vandv.predictor_reporting import report_prediction_as_graphs_html


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3), text_header='abstract',
                 cached_folder_name=None, max_df=0.1, user_ngrams=None, prefilter_terms=0,
                 terms_threshold=None, output_name=None, calculate_timeseries=None, m_steps_ahead=5,
                 emergence_index='porter', exponential=False, nterms=50, patents_per_quarter_threshold=20, sma=None):

        # load data
        self.__data_filename = data_filename
        self.__date_dict = docs_mask_dict['date']
        self.__timeseries_date_dict = docs_mask_dict['timeseries_date']
        self.__timeseries_data = []

        self.__emergence_list = []
        self.__pick_method = pick_method
        # calculate or fetch tf-idf mat
        if cached_folder_name is None:
            dataframe = data_factory.get(data_filename)
            utils.checkdf(dataframe, calculate_timeseries, docs_mask_dict, text_header)
            utils.remove_empty_documents(dataframe, text_header)

            self.__tfidf_obj = tfidf_from_text(text_series=dataframe[text_header],
                                               ngram_range=ngram_range,
                                               max_document_frequency=max_df,
                                               tokenizer=LemmaTokenizer())
            tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=ngram_range, uni_factor=0.8, unbias=True)
            self.__tfidf_obj.apply_weights(tfidf_mask_obj.tfidf_mask)

            if prefilter_terms != 0:
                tfidf_reduce_obj = TfidfReduce(self.__tfidf_obj.tfidf_matrix, self.__tfidf_obj.feature_names)
                term_score_tuples = tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)
                num_tuples_to_retain = min(prefilter_terms, len(term_score_tuples))

                feature_subset = sorted([x[1] for x in term_score_tuples[:num_tuples_to_retain]])

                number_of_ngrams_before = len(self.__tfidf_obj.feature_names)
                self.__tfidf_obj = tfidf_subset_from_features(self.__tfidf_obj, feature_subset)
                number_of_ngrams_after = len(self.__tfidf_obj.feature_names)
                print(f'Reduced number of terms by pre-filtering from {number_of_ngrams_before:,} '
                      f'to {number_of_ngrams_after:,}')

            self.__cpc_dict = utils.cpc_dict(dataframe)
            if docs_mask_dict['date_header'] is None:
                self.__cached_folder_name = path.join('cached', output_name + f'-mdf-{max_df}')
                self.__dates = None
            else:
                self.__dates = scripts.utils.date_utils.generate_year_week_dates(dataframe,
                                                                                 docs_mask_dict['date_header'])
                min_date = min(self.__dates)
                max_date = max(self.__dates)
                self.__cached_folder_name = path.join('cached', output_name + f'-mdf-{max_df}-{min_date}-{max_date}')

            utils.pickle_object('tfidf', self.__tfidf_obj, self.__cached_folder_name)
            utils.pickle_object('dates', self.__dates, self.__cached_folder_name)
            utils.pickle_object('cpc_dict', self.__cpc_dict, self.__cached_folder_name)

        else:
            print(f'Reading document and TFIDF from pickle {cached_folder_name}')

            self.__cached_folder_name = path.join('cached', cached_folder_name)
            self.__tfidf_obj = utils.unpickle_object('tfidf', self.__cached_folder_name)
            self.__dates = utils.unpickle_object('dates', self.__cached_folder_name)
            self.__cpc_dict = utils.unpickle_object('cpc_dict', self.__cached_folder_name)

            if self.__dates is not None:
                min_date = min(self.__dates)
                max_date = max(self.__dates)
                print(f'Document year-week dates range from {min_date // 100}-{(min_date % 100):02d} '
                      f'to {max_date // 100}-{(max_date % 100):02d}')

            WordAnalyzer.init(
                tokenizer=LemmaTokenizer(),
                preprocess=lowercase_strip_accents_and_ownership,
                ngram_range=ngram_range)

        # todo: pipeline is now a one-way trip of data, slowly collapsing / shrinking it as we don't need to keep
        #  the original. We're really just filtering down.

        # todo: build up a list of functions to apply as document filters. all filters to have common args (c/o
        #  partialfunc if required) so we can then call them in sequence...
        #  from a combiner.
        #  each func just returns an array of bool (or 0/1)
        #  if union - create union combiner, else create intersection combiner. combiner = union if... else intersection
        #  weights = combiner(list of funcs, data set)
        #  combiner: if list is empty, return [1] * size; if single entry, return its array
        #  union: if more entries after single, add / or
        #  intersection: if more entries after single, multiple / and
        #  then apply mask to tfidf object and df (i.e. remove rows with false or 0); do this in place
        print(f'Applying documents filter...')
        # docs weights( column, dates subset + time, citations etc.)
        doc_filters = DocumentsFilter(self.__dates, docs_mask_dict, self.__cpc_dict,
                                      self.__tfidf_obj.tfidf_matrix.shape[0]).doc_filters

        # todo: build up list of weight functions (left with single remaining arg etc via partialfunc)
        #  combine(list, tfidf) => multiplies weights together, then multiplies across tfidf (if empty, no side effect)

        # todo: this is another weight function...

        # term weights - embeddings
        print(f'Applying terms filter...')
        filter_terms_obj = FilterTerms(self.__tfidf_obj.feature_names, user_ngrams, threshold=terms_threshold)
        term_weights = filter_terms_obj.ngram_weights_vec

        # todo: replace tfidf_mask with isolated functions: clean_unigrams, unbias_ngrams;
        #  these operate directly on tfidf
        #  Hence return nothing - operate in place on tfidf.
        print(f'Creating a masked tfidf matrix from filters...')
        # tfidf mask ( doc_ids, doc_weights, embeddings_filter will all merge to a single mask in the future)
        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=ngram_range, uni_factor=0.8)
        tfidf_mask_obj.update_mask(doc_filters, term_weights)
        tfidf_mask = tfidf_mask_obj.tfidf_mask

        # todo: this mutiply and remove null will disappear - maybe put weight combiner last so it can remove 0 weights
        # mask the tfidf matrix

        tfidf_masked = tfidf_mask.multiply(self.__tfidf_obj.tfidf_matrix)

        tfidf_masked, self.__dates = utils.remove_all_null_rows_global(tfidf_masked, self.__dates)
        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,}'
              f' / {self.__tfidf_obj.tfidf_matrix.shape[0]:,} documents')

        # todo: no advantage in classes - just create term_count and extract_ngrams as functions

        self.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        self.__timeseries_data = None

        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)
        self.__term_score_tuples = utils.stop_tup(self.__term_score_tuples, WordAnalyzer.stemmed_stop_word_set_uni,
                                                  WordAnalyzer.stemmed_stop_word_set_n)

        # todo: no output method; just if statements to call output functions...?
        #  Only supply what they each directly require

        # todo: hence Pipeline then becomes a single function
        if not calculate_timeseries:
            return

        # TODO: offer timeseries cache as an option. Then filter dates and terms after reading the cached matrix
        print(f'Creating timeseries matrix...')
        if cached_folder_name is None or not (
                path.isfile(utils.pickle_name('weekly_series_terms', self.__cached_folder_name))
                and path.isfile(utils.pickle_name('weekly_series_global', self.__cached_folder_name))
                and path.isfile(utils.pickle_name('weekly_isodates', self.__cached_folder_name))):
            self.__timeseries_data = self.__tfidf_reduce_obj.create_timeseries_data(self.__dates)
            [self.__term_counts_per_week, self.__term_ngrams, self.__number_of_patents_per_week,
             self.__weekly_iso_dates] = self.__timeseries_data

            utils.pickle_object('weekly_series_terms', self.__term_counts_per_week, self.__cached_folder_name)
            utils.pickle_object('weekly_series_global', self.__number_of_patents_per_week, self.__cached_folder_name)
            utils.pickle_object('weekly_isodates', self.__weekly_iso_dates, self.__cached_folder_name)
        else:
            self.__term_counts_per_week = utils.unpickle_object('weekly_series_terms', self.__cached_folder_name)
            self.__number_of_patents_per_week = utils.unpickle_object('weekly_series_global', self.__cached_folder_name)
            self.__weekly_iso_dates = utils.unpickle_object('weekly_isodates', self.__cached_folder_name)
            self.__term_ngrams = self.__tfidf_obj.feature_names

        self.__M = m_steps_ahead

        # TODO: define period from command line, then cascade through the code

        term_counts_per_week_csc = self.__term_counts_per_week.tocsc()
        self.__timeseries_quarterly = []
        self.__timeseries_intercept = []
        self.__timeseries_derivatives = []
        self.__timeseries_quarterly_smoothed = []
        self.__term_nonzero_dates = []

        all_quarters, all_quarterly_values = self.__x = scripts.utils.date_utils.timeseries_weekly_to_quarterly(
            self.__weekly_iso_dates, self.__number_of_patents_per_week)

        # find indexes for date-range
        min_date = max_date = None
        if self.__timeseries_date_dict is not None:
            min_date = self.__timeseries_date_dict['from']
            max_date = self.__timeseries_date_dict['to']

        min_i = 0
        max_i = len(all_quarters)

        for i, quarter in enumerate(all_quarters):
            if min_date is not None and min_date < quarter:
                break
            min_i = i

        for i, quarter in enumerate(all_quarters):
            if max_date is not None and max_date < quarter:
                break
            max_i = i
        self.__lims = [min_i, max_i]
        self.__timeseries_quarterly_smoothed = None if sma is None else []

        for term_index in tqdm(range(self.__term_counts_per_week.shape[1]), unit='term',
                               desc='Calculating  quarterly timeseries',
                               leave=False, unit_scale=True):
            row_indices, row_values = utils.get_row_indices_and_values(term_counts_per_week_csc, term_index)
            weekly_iso_dates = [self.__weekly_iso_dates[x] for x in row_indices]
            non_zero_dates, quarterly_values = scripts.utils.date_utils.timeseries_weekly_to_quarterly(weekly_iso_dates,
                                                                                                       row_values)
            non_zero_dates, quarterly_values = utils.fill_missing_zeros(quarterly_values, non_zero_dates, all_quarters)
            self.__timeseries_quarterly.append(quarterly_values)

        if emergence_index == 'gradients' or sma == 'kalman':
            if cached_folder_name is None or not (
                    path.isfile(utils.pickle_name('smooth_series_s', self.__cached_folder_name))
                    and path.isfile(utils.pickle_name('derivatives', self.__cached_folder_name))):
                for term_index, quarterly_values in tqdm(enumerate(self.__timeseries_quarterly), unit='term',
                                                         desc='smoothing quarterly timeseries with kalman filter',
                                                         leave=False, unit_scale=True,
                                                         total=len(self.__timeseries_quarterly)):
                    _, _1, smooth_series_s, _intercept = StateSpaceModel(quarterly_values).run_smoothing()

                    smooth_series = smooth_series_s[0].tolist()[0]
                    smooth_series_no_negatives = np.clip(smooth_series, a_min=0, a_max=None)
                    self.__timeseries_quarterly_smoothed.append(smooth_series_no_negatives.tolist())

                    derivatives = smooth_series_s[1].tolist()[0]
                    self.__timeseries_derivatives.append(derivatives)

                utils.pickle_object('smooth_series_s', self.__timeseries_quarterly_smoothed, self.__cached_folder_name)
                utils.pickle_object('derivatives', self.__timeseries_derivatives, self.__cached_folder_name)

            else:
                self.__timeseries_quarterly_smoothed = utils.unpickle_object('smooth_series_s',
                                                                             self.__cached_folder_name)
                self.__timeseries_derivatives = utils.unpickle_object('derivatives', self.__cached_folder_name)

        if sma == 'savgol':
            for quarterly_values in tqdm(self.__timeseries_quarterly, unit='term',
                                         desc='savgol smoothing quarterly timeseries',
                                         leave=False, unit_scale=True):
                smooth_series = savgol_filter(quarterly_values, 9, 2, mode='nearest')
                smooth_series_no_negatives = np.clip(smooth_series, a_min=0, a_max=None)
                self.__timeseries_quarterly_smoothed.append(smooth_series_no_negatives.tolist())

        em = Emergence(all_quarterly_values[min_i:max_i])
        for term_index in tqdm(range(self.__term_counts_per_week.shape[1]), unit='term', desc='Calculating eScore',
                               leave=False, unit_scale=True):
            if term_weights[term_index] == 0.0:
                continue
            term_ngram = self.__term_ngrams[term_index]

            if self.__timeseries_quarterly_smoothed is not None:
                quarterly_values = list(self.__timeseries_quarterly_smoothed[term_index])[min_i:max_i]
            else:
                quarterly_values = list(self.__timeseries_quarterly[term_index])[min_i:max_i]

            if len(quarterly_values) == 0 or max(list(self.__timeseries_quarterly[term_index][min_i:max_i])) < float(
                    patents_per_quarter_threshold):
                continue

            if emergence_index == 'quadratic':
                escore = em.escore2(quarterly_values)
            elif emergence_index == 'porter':
                if not em.is_emergence_candidate(quarterly_values):
                    continue
                escore = em.calculate_escore(quarterly_values)
            elif emergence_index == 'gradients':
                derivatives = self.__timeseries_derivatives[term_index][min_i:max_i]
                escore = em.net_growth(quarterly_values, derivatives)
            else:
                weekly_values = term_counts_per_week_csc.getcol(term_index).todense().ravel().tolist()[0]
                escore = em.escore_exponential(weekly_values)

            self.__emergence_list.append((term_ngram, escore))

        nterms2 = min(nterms, len(self.__emergence_list))
        self.__emergence_list.sort(key=lambda emergence: -emergence[1])

        self.__emergent = [x[0] for x in self.__emergence_list[:nterms2]]
        self.__declining = [x[0] for x in self.__emergence_list[-nterms2:]]
        self.__declining.reverse()
        self.__stationary = [x[0] for x in utils.stationary_terms(self.__emergence_list, nterms2)]

    @staticmethod
    def label_prediction_simple(values):
        if np.isnan(values).any() or np.isinf(values).any():
            return 'NaN'
        x = np.array(range(len(values))).reshape((-1, 1))
        model = LinearRegression().fit(x, values)
        slope = model.coef_

        if slope > 0.1:
            return 'emerging'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'level'

    def label_prediction(self, derivatives):
        if np.isnan(derivatives).any() or np.isinf(derivatives).any():
            return 'NaN'
        sum_derivatives = sum(derivatives)

        x = np.array(range(len(derivatives))).reshape((-1, 1))
        model = LinearRegression().fit(x, derivatives)
        derivative_slope = model.coef_

        if sum_derivatives > 0.1:
            if derivative_slope > 0:
                return 'p-increase'
            else:
                return 't-increase'
        elif sum_derivatives < -0.1:
            if derivative_slope > 0:
                return 't-decrease'
            else:
                return 'p-decrease'
        else:
            return 'level'

    def evaluate_predictions(self, timeseries, test_terms, term_ngrams, methods,smooth_series=None, derivatives=None, window=30, min_k=3, max_k=8):

        results_term = {}
        window_interval=1

        for test_term in tqdm(test_terms, unit='term', unit_scale=True):
            term_index = term_ngrams.index(test_term)
            term_series = timeseries[term_index]
            term_series_smooth = smooth_series[term_index]
            nperiods = len(term_series)
            num_runs = nperiods - window

            results_method = {}
            for method in methods:
                scores = {}
                for i in range(0, num_runs, window_interval):

                    term_series_window = term_series[i:i + window]

                    term_series_window = np.clip(term_series_window, 0.00001, None)
                    smooth_series_window = np.clip(term_series_smooth[i:i + window], 0.00001, None)
                    history_series = term_series_window[:-max_k]
                    test_series = smooth_series_window[-max_k:]

                    factory = PredictorFactory.predictor_factory(method, '', history_series, max_k)
                    predicted_term_values = factory.predict_counts()

                    if derivatives is not None:
                        term_derivatives = derivatives[term_index]
                        term_derivatives_window = term_derivatives[i:i + window]
                        test_derivatives = term_derivatives_window[-max_k:]
                        predicted_derivatives = factory.predict_derivatives()

                    for num_periods in range(min_k, max_k, 2):
                        predict_func = self.label_prediction_simple if derivatives is None else self.label_prediction
                        values = predicted_term_values if derivatives is None else predicted_derivatives
                        test_values = test_series if derivatives is None else test_derivatives

                        predicted_label = predict_func(values[:num_periods])
                        actual_label = predict_func(test_values[:num_periods])
                        score = 1 if predicted_label == actual_label else 0
                        # print(test_term +'_'+method+ '_'+actual_label+'_' + predicted_label + '_' + str(i))
                        if num_periods in scores:
                            scores[num_periods] += score
                        else:
                            scores[num_periods] = score

                for num_periods in range(min_k, max_k, 2):
                    results_method[method+'_' + str(num_periods)] = scores[num_periods]/(num_runs/window_interval)

            results_term[test_term] = results_method
        return results_term

    @staticmethod
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)

    def evaluate_state_space_pred(self, timeseries, derivatives, test_terms, term_ngrams, window=20,
                                  k_range=(2, 3, 4, 5)):

        results = {}

        for test_term in tqdm(test_terms, unit='term', unit_scale=True):
            term_index = term_ngrams.index(test_term)
            series = timeseries[term_index]
            term_derivatives = derivatives[term_index]
            nperiods = len(series)
            num_runs = nperiods - window
            results[test_term] = {}
            for k in k_range:
                score = 0
                results_for_k = {}

                for i in range(num_runs):
                    alpha, mse = StateSpaceModel(series[i:i + window]).run_smooth_forecast(k=k)
                    results_for_k[i] = {}

                    predicted_term_values = np.array(alpha[0])[0]
                    results_for_k[i]['predicted_values'] = predicted_term_values

                    predicted_term_derivatives = np.array(alpha[1])[0]
                    results_for_k[i]['predicted_derivative'] = predicted_term_derivatives
                    results_for_k[i]['predicted_label'] = self.label_prediction(predicted_term_derivatives, k=k)

                    if num_runs > 1:
                        results_for_k[i]['derivative'] = derivatives[i + window - k:i + window]
                        results_for_k[i]['label'] = self.label_prediction(
                            np.array(term_derivatives[i + window - k:i + window]), k=k)
                        score += (results_for_k[i]['label'] == results_for_k[i]['predicted_label'])

                if num_runs > 1:
                    results_for_k['accuracy'] = score / num_runs

                results[test_term][k] = results_for_k
        return results

    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50, n_nmf_topics=0):
        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, emergence_list=self.__emergence_list,
                                  wordcloud_title=wordcloud_title, tfidf_reduce_obj=self.__tfidf_reduce_obj,
                                  name=outname, nterms=nterms, timeseries_data=self.__timeseries_data,
                                  date_dict=self.__date_dict, pick=self.__pick_method,
                                  doc_pickle_file_name=self.__data_filename, nmf_topics=n_nmf_topics)

    @property
    def term_score_tuples(self):
        return self.__term_score_tuples

    def get_multiplot(self, timeseries_terms_smooth, timeseries, test_terms, term_ngrams, lims, method='Net Growth',
                      category='emergent'):

        if len(test_terms) != 30:
            raise ValueError('Only supports 30 terms as multiplot is 6x5')

        # libraries and data
        import matplotlib.pyplot as plt
        import pandas as pd

        series_dict = {'x': range(len(timeseries[0]))}
        for test_term in test_terms:
            term_index = term_ngrams.index(test_term)
            series_dict[term_ngrams[term_index]] = timeseries[term_index]

        series_dict_smooth = {'x': range(len(timeseries_terms_smooth[0]))}
        for test_term in test_terms:
            term_index = term_ngrams.index(test_term)
            series_dict_smooth[term_ngrams[term_index]] = timeseries_terms_smooth[term_index]

        # make a data frame
        df = pd.DataFrame(series_dict)
        df_smooth = pd.DataFrame(series_dict_smooth)

        # initialize the figure
        plt.style.use('seaborn-darkgrid')

        # create a color palette

        # multiple line plot
        num = 0
        for column in df.drop('x', axis=1):
            num += 1

            # find the right spot on the plot
            plt.subplot(6, 5, num)

            # plot the lineplot
            plt.plot(df['x'], df[column], color='b', marker='', linewidth=1.4, alpha=0.9, label=column)
            plt.plot(df['x'], df_smooth[column], color='g', linestyle='-', marker='', label='smoothed ground truth')

            plt.axvline(x=lims[0], color='k', linestyle='--')
            plt.axvline(x=lims[1], color='k', linestyle='--')

            # same limits for everybody!
            plt.xlim(0, series_dict['x'])

            # not ticks everywhere
            if num in range(26):
                plt.tick_params(labelbottom='off')

            # plt.tick_params(labelleft='off')

            # add title
            plt.title(column, loc='left', fontsize=12, fontweight=0)

        # general title
        plt.suptitle(category + " keywords selection using the " + method + " index", fontsize=13, fontweight=0,
                     color='black', style='italic')

        # axis title
        plt.show()

    @property
    def timeseries_data(self):
        return self.__timeseries_data

    def run(self, predictors_to_run, emergence, normalized=False, train_test=False, ss_only=True):
        if emergence == 'emergent':
            terms = self.__emergent
        elif emergence == 'stationary':
            terms = self.__stationary
        elif emergence == 'declining':
            terms = self.__declining
        else:
            raise ValueError(f'Unrecognised value for emergence_type: {emergence}')

        if len(terms) == 0:
            print(f'Analysis of {emergence} failed as no terms were detected,'
                  f' likely because -mpq is too large for dataset provided')
            return '', None

        html_results = ''

        if ss_only:
            # self.get_state_space_forecast(self.__timeseries_quarterly, self.__emergent, self.__term_ngrams)
            if train_test:
                k_range = range(2, self.__M + 1)
                window_size = 30
            else:
                k_range = [self.__M]
                window_size = len(self.__timeseries_quarterly[0]) - 1

            # results = self.evaluate_state_space_pred(self.__timeseries_quarterly, self.__timeseries_derivatives,
            #                                          terms, self.__term_ngrams, window=window_size)

            results = self.evaluate_predictions(self.__timeseries_quarterly, terms, self.__term_ngrams,
                                                predictors_to_run,
                                                smooth_series=self.__timeseries_quarterly_smoothed, window=window_size)

            # results = self.evaluate_predictions(self.__timeseries_quarterly_smoothed, terms, self.__term_ngrams,
            #                                     predictors_to_run, window=window_size)
            print(results)

            # utils.pickle_object('results', results, self.__cached_folder_name)

            html_results += f'<h2>State Space Model: {emergence} terms</h2>\n'
            html_results += f'<p>Window size: {window_size}</p>\n'

            if train_test:
                html_results += '<p><b>Testing predictions</b></p>\n'
                html_results += f'<h3>Term analysis</h2>\n'
                html_results += ssm_reporting.html_table(results)
                html_results += f'<h3>Analysis summary</h2>\n'
                html_results += ssm_reporting.summary_html_table(results)

            return html_results, None
        else:
            results, training_values, test_values, smoothed_training_values, smoothed_test_values = evaluate_prediction(
                self.__timeseries_quarterly, self.__term_ngrams, predictors_to_run, test_terms=terms,
                test_forecasts=train_test, timeseries_all=self.__number_of_patents_per_week if normalized else None,
                num_prediction_periods=self.__M, smoothed_series=self.__timeseries_quarterly_smoothed)

            predicted_emergence = map_prediction_to_emergence_label(results, smoothed_training_values, smoothed_test_values,
                                                                    predictors_to_run, test_terms=terms)

            html_results += report_predicted_emergence_labels_html(predicted_emergence)

            html_results += report_prediction_as_graphs_html(results, predictors_to_run, self.__weekly_iso_dates,
                                                             test_values=test_values,
                                                             smoothed_test_values=smoothed_test_values,
                                                             test_terms=terms, training_values=training_values,
                                                             smoothed_training_values=smoothed_training_values,
                                                             normalised=normalized,
                                                             test_forecasts=train_test, lims=self.__lims)

            return html_results, training_values.items()
