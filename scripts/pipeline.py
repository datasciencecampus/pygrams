import bz2
import pickle
from os import makedirs, path

from pandas import read_pickle
from tqdm import tqdm

import scripts.data_factory as datafactory
import scripts.output_factory as output_factory
from scripts.algorithms.emergence import Emergence
from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import TFIDF
from scripts.utils import utils
from scripts.vandv.emergence_labels import map_prediction_to_emergence_label, report_predicted_emergence_labels_html
from scripts.vandv.graphs import report_prediction_as_graphs_html
from scripts.vandv.predictor import evaluate_prediction


def checkdf( df, emtec, docs_mask_dict, text_header):
    app_exit = False

    if emtec or docs_mask_dict['time'] or docs_mask_dict['date'] is not None:
        if docs_mask_dict['date_header'] not in df.columns:
            print(f"date_header '{docs_mask_dict['date_header']}' not in dataframe")
            app_exit = True

    if text_header not in df.columns:
        print(f"text_header '{text_header}' not in dataframe")
        app_exit = True

    if app_exit:
        exit(0)


def remove_empty_documents(data_frame, text_header):
    num_docs_before_sift = data_frame.shape[0]
    data_frame.dropna(subset=[text_header], inplace=True)
    num_docs_after_sift = data_frame.shape[0]
    num_docs_sifted = num_docs_before_sift - num_docs_after_sift
    print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3),
                 normalize_rows=False, text_header='abstract', term_counts=False,
                 pickled_tf_idf_file_name=None, max_df=0.1, user_ngrams=None,
                 output_name=None, emerging_technology=None):

        # load data
        self.__data_filename = data_filename
        self.__date_dict = docs_mask_dict['date']
        self.__time = docs_mask_dict['time']

        self.__pick_method = pick_method
        # calculate or fetch tf-idf mat
        if pickled_tf_idf_file_name is None:

            self.__dataframe = datafactory.get(data_filename)
            checkdf(self.__dataframe, emerging_technology, docs_mask_dict, text_header)

            remove_empty_documents(self.__dataframe, text_header)
            self.__tfidf_obj = TFIDF(text_series=self.__dataframe[text_header], ngram_range=ngram_range,
                                     max_document_frequency=max_df, tokenizer=LemmaTokenizer())

            self.__text_lengths = self.__dataframe[text_header].map(len).tolist()
            self.__dataframe.drop(columns=[text_header], inplace=True)

            tfidf_filename = path.join('outputs', 'tfidf', output_name + '-tfidf.pkl.bz2')
            makedirs(path.dirname(tfidf_filename), exist_ok=True)
            with bz2.BZ2File(tfidf_filename, 'wb') as pickle_file:
                pickle.dump(
                    (self.__tfidf_obj, self.__dataframe, self.__text_lengths),
                    pickle_file,
                    protocol=4)

        else:
            print(f'Reading document and TFIDF from pickle {pickled_tf_idf_file_name}')
            self.__tfidf_obj, self.__dataframe, self.__text_lengths = read_pickle(pickled_tf_idf_file_name)

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

        # docs weights( column, dates subset + time, citations etc.)
        doc_filters = DocumentsFilter(self.__dataframe, docs_mask_dict).doc_weights

        # todo: build up list of weight functions (left with single remaining arg etc via partialfunc)
        #  combine(list, tfidf) => multiplies weights together, then multiplies across tfidf (if empty, no side effect)

        doc_weights = DocumentsWeights(self.__dataframe, docs_mask_dict['time'], docs_mask_dict['cite'],
                                       docs_mask_dict['date_header'], self.__text_lengths,
                                       norm_rows=normalize_rows).weights
        doc_weights = [a * b for a, b in zip(doc_filters, doc_weights)]

        # todo: this is another weight function...

        # term weights - embeddings
        filter_terms_obj = FilterTerms(self.__tfidf_obj.feature_names, user_ngrams, threshold=0.75)
        term_weights = filter_terms_obj.ngram_weights_vec

        # todo: replace tfidf_mask with isolated functions: clean_unigrams, unbias_ngrams;
        #  these operate directly on tfidf
        #  Hence return nothing - operate in place on tfidf.

        # tfidf mask ( doc_ids, doc_weights, embeddings_filter will all merge to a single mask in the future)
        tfidf_mask_obj = TfidfMask(self.__tfidf_obj, ngram_range=ngram_range, uni_factor=0.8)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        tfidf_mask = tfidf_mask_obj.tfidf_mask

        # todo: this mutiply and remove null will disappear - maybe put weight combiner last so it can remove 0 weights
        # mask the tfidf matrix
        tfidf_matrix = self.__tfidf_obj.tfidf_matrix
        tfidf_masked = tfidf_mask.multiply(tfidf_matrix)

        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,} / {tfidf_matrix.shape[0]:,} documents')

        # todo: no advantage in classes - just create term_count and extract_ngrams as functions

        self.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        self.__term_counts_data = None
        if term_counts or emerging_technology:
            self.__term_counts_data = self.__tfidf_reduce_obj.create_terms_count(self.__dataframe,
                                                                                 docs_mask_dict['date_header'])
        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)

        # todo: no output method; just if statements to call output functions...?
        #  Only supply what they each directly require

        # todo: hence Pipeline then becomes a single function

    @property
    def term_counts_data(self):
        return self.__term_counts_data

    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50):

        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, wordcloud_title=wordcloud_title,
                                  tfidf_reduce_obj=self.__tfidf_reduce_obj, name=outname,
                                  nterms=nterms, term_counts_data=self.__term_counts_data,
                                  date_dict=self.__date_dict, pick=self.__pick_method,
                                  doc_pickle_file_name=self.__data_filename, time=self.__time)


    @property
    def term_score_tuples(self):
        return self.__term_score_tuples


# 'USPTO-granted-full-random-500000-term_counts.pkl.bz2'
class PipelineEmtech(object):
    def __init__(self, term_counts_data, m_steps_ahead=5, curves=True, nterms=50, minimum_patents_per_quarter=20,
                 outname=None):
        self.__M = m_steps_ahead

        [self.__term_counts_per_week, self.__term_ngrams, self.__number_of_patents_per_week,
         self.__weekly_iso_dates] = term_counts_data

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

        if len(self.__emergence_list) == 0:
            self.__emergent = []
            self.__declining = []
            self.__stationary = []
            return

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
        filename_and_path = path.join('outputs', 'reports', outname + '_emergence.txt')
        with open(filename_and_path, 'w') as file:
            print()
            print('Emergent')
            file.write('Emergent\n')
            for tup in self.__emergence_list[:nterms]:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1])+ '\n')
            print()
            file.write('\n')
            print('Stationary')
            file.write('Stationary\n')
            for tup in self.__emergence_list[stationary_start_index:stationary_end_index]:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1]) + '\n')
            print()
            file.write('\n')

            print('Declining')
            file.write('Declining'+ '\n')
            for tup in self.__emergence_list[-nterms:]:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1])+ '\n')
            print()
            file.write('\n')
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

        if len(terms) == 0:
            print(f'Analysis of {emergence} failed as no terms were detected,'
                  f' likely because -mpq is too large for dataset provided')
            return

        html_results = ''

        results, training_values, test_values = evaluate_prediction(self.__term_counts_per_week, self.__term_ngrams,
                                                                    predictors_to_run, self.__weekly_iso_dates,
                                                                    test_terms=terms, test_forecasts=train_test,
                                                                    normalised=normalized,
                                                                    number_of_patents_per_week=self.__number_of_patents_per_week,
                                                                    num_prediction_periods=self.__M)

        predicted_emergence = map_prediction_to_emergence_label(results, training_values, test_values,
                                                                predictors_to_run, test_terms=terms)

        html_results += report_predicted_emergence_labels_html(predicted_emergence)

        html_results += report_prediction_as_graphs_html(results, predictors_to_run, self.__weekly_iso_dates,
                                                         test_values=test_values,
                                                         test_terms=terms, training_values=training_values,
                                                         normalised=normalized,
                                                         test_forecasts=train_test)

        return html_results
