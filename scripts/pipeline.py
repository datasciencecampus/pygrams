import os

from pandas import read_pickle

from tqdm import tqdm

import scripts.data_factory as datafactory
import scripts.output_factory as output_factory
from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import TFIDF
from scripts.utils import utils
from scripts.utils.date_utils import year2pandas_earliest_date, year2pandas_latest_date
from scripts.algorithms.emergence import Emergence
from scripts.vandv.emergence_labels import map_prediction_to_emergence_label, report_predicted_emergence_labels_html
from scripts.vandv.graphs import report_prediction_as_graphs_html
from scripts.vandv.predictor import evaluate_prediction


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3),
                 normalize_rows=False, text_header='abstract', term_counts=False,
                 pickled_tf_idf=None, max_df=0.1, user_ngrams=None):


        # load data
        self.__data_filename = data_filename

        doc_dates = docs_mask_dict['dates']
        year_from = doc_dates[0]
        year_to = doc_dates[1]
        month_from = doc_dates[2]
        month_to = doc_dates[3]

        date_from = year2pandas_earliest_date(year_from, month_from)
        date_to = year2pandas_latest_date(year_to, month_to)
        self.__date_range = [date_from, date_to]
        self.__time = docs_mask_dict['time']

        self.__pick_method = pick_method
        # calculate or fetch tf-idf mat
        if pickled_tf_idf is None:
            df = datafactory.get(data_filename)
            self.__tfidf_obj = TFIDF(docs_df=df, ngram_range=ngram_range, max_document_frequency=max_df,
                                     tokenizer=LemmaTokenizer(), text_header=text_header)
        else:
            print(f'Reading document and TFIDF from pickle {pickled_tf_idf}')
            self.__tfidf_obj = read_pickle(pickled_tf_idf)
            df = self.__tfidf_obj.dataframe

        # docs weights( column, dates subset + time, citations etc.)
        doc_filters = DocumentsFilter(df, docs_mask_dict).doc_weights
        doc_weights = DocumentsWeights(df, docs_mask_dict['time'], docs_mask_dict['cite'],
                                       docs_mask_dict['dates'][-1], text_header=text_header,
                                       norm_rows=normalize_rows).weights
        doc_weights = [a * b for a, b in zip(doc_filters, doc_weights)]

        # term weights - embeddings
        filter_terms_obj = FilterTerms(self.__tfidf_obj.feature_names, user_ngrams,
                                       file_name=os.path.join('data', 'embeddings', 'glove', 'w2v_glove.6B.50d.txt'))
        term_weights = filter_terms_obj.ngram_weights_vec

        # tfidf mask ( doc_ids, doc_weights, embeddings_filter will all merge to a single mask in the future)
        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=ngram_range, uni_factor=0.8)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        tfidf_mask = tfidf_mask_obj.tfidf_mask

        # mask the tfidf matrix
        tfidf_matrix = self.__tfidf_obj.tfidf_matrix
        tfidf_masked = tfidf_mask.multiply(tfidf_matrix)
        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,} / {tfidf_matrix.shape[0]:,} documents')

        self.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        self.__term_counts_data = None
        if term_counts:
            self.__term_counts_data = self.__tfidf_reduce_obj.create_terms_count(df, docs_mask_dict['dates'][-1])
        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)


    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50):

        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, wordcloud_title=wordcloud_title,
                                  tfidf_reduce_obj=self.__tfidf_reduce_obj, name=outname,
                                  nterms=nterms, term_counts_data=self.__term_counts_data,
                                  tfidf_obj=self.__tfidf_obj, date_range=self.__date_range, pick=self.__pick_method,
                                  doc_pickle_file_name=self.__data_filename, time=self.__time)

    @property
    def term_score_tuples(self):
        return self.__term_score_tuples

# 'USPTO-granted-full-random-500000-term_counts.pkl.bz2'
class PipelineEmtech(object):
    def __init__(self, term_counts_filename, m_steps_ahead=5, curves=True, nterms=50, minimum_patents_per_quarter=20):
        self.__M = m_steps_ahead

        [self.__term_counts_per_week, self.__term_ngrams, self.__number_of_patents_per_week,
         self.__weekly_iso_dates] = read_pickle(
            os.path.join(term_counts_filename))
        self.__output_folder = os.path.join('outputs', 'emtech')
        self.__base_file_name = os.path.basename(term_counts_filename)

        term_counts_per_week_csc = self.__term_counts_per_week.tocsc()

        em = Emergence(self.__number_of_patents_per_week)
        self.__emergence_list = []
        for term_index in tqdm(range(self.__term_counts_per_week.shape[1]), unit='term', desc='Calculating eScore',
                               leave=False, unit_scale=True):
            term_ngram = self.__term_ngrams[term_index]
            row_indices, row_values = utils.get_row_indices_and_values(term_counts_per_week_csc, term_index)

            if len(row_values) == 0:
                continue

            weekly_iso_dates = [self.__weekly_iso_dates[x] for x in row_indices]

            _, quarterly_values = utils.timeseries_weekly_to_quarterly(weekly_iso_dates, row_values)
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
