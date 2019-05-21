import bz2
import pickle
from os import makedirs, path

from pandas import read_pickle
from tqdm import tqdm

import scripts.data_factory as datafactory
import scripts.output_factory as output_factory
import scripts.utils.date_utils
from scripts.algorithms.emergence import Emergence
from scripts.documents_filter import DocumentsFilter
from scripts.filter_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer, WordAnalyzer, lowercase_strip_accents_and_ownership
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import tfidf_subset_from_features, tfidf_from_text
from scripts.utils import utils
from scripts.vandv.emergence_labels import map_prediction_to_emergence_label, report_predicted_emergence_labels_html
from scripts.vandv.graphs import report_prediction_as_graphs_html
from scripts.vandv.predictor import evaluate_prediction


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3), text_header='abstract',
                 term_counts=False, pickled_base_file_name=None, max_df=0.1, user_ngrams=None, prefilter_terms=0,
                 terms_threshold=None, output_name=None, emerging_technology=None):

        # load data
        self.__data_filename = data_filename
        self.__date_dict = docs_mask_dict['date']
        self.__time = docs_mask_dict['time']

        self.__pick_method = pick_method
        # calculate or fetch tf-idf mat
        if pickled_base_file_name is None:

            dataframe = datafactory.get(data_filename)
            utils.checkdf(dataframe, emerging_technology, docs_mask_dict, text_header, term_counts)
            utils.remove_empty_documents(dataframe, text_header)

            self.__tfidf_obj = tfidf_from_text(text_series=dataframe[text_header],
                                               ngram_range=ngram_range,
                                               max_document_frequency=max_df,
                                               tokenizer=LemmaTokenizer())

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
            self.__dates = scripts.utils.date_utils.generate_year_week_dates(dataframe, docs_mask_dict['date_header'])

            base_pickle_path = path.join('outputs', 'tfidf')
            makedirs(base_pickle_path, exist_ok=True)

            def pickle_object(short_name, obj):
                file_name = path.join(base_pickle_path, output_name + f'-mdf-{max_df}-{short_name}.pkl.bz2')
                with bz2.BZ2File(file_name, 'wb') as pickle_file:
                    pickle.dump(obj, pickle_file, protocol=4, fix_imports=False)

            pickle_object('tfidf', self.__tfidf_obj)
            pickle_object('dates', self.__dates)
            pickle_object('cpc_dict', self.__cpc_dict)

        else:
            print(f'Reading document and TFIDF from pickle {pickled_base_file_name}')
            self.__tfidf_obj = read_pickle(pickled_base_file_name + '-tfidf.pkl.bz2')
            self.__dates = read_pickle(pickled_base_file_name + '-dates.pkl.bz2')
            self.__cpc_dict = read_pickle(pickled_base_file_name + '-cpc_dict.pkl.bz2')

            if self.__dates is not None:
                min_date = min(self.__dates)
                max_date = max(self.__dates)
                print(f'Document year-week dates range from {min_date/100}-{min_date%100} to {max_date/100}-{max_date%100}')

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
        print(f'Aplying terms filter...')
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
        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,} / {self.__tfidf_obj.tfidf_matrix.shape[0]:,} documents')

        # todo: no advantage in classes - just create term_count and extract_ngrams as functions

        self.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, self.__tfidf_obj.feature_names)
        self.__term_counts_data = None
        if term_counts or emerging_technology:
            print(f'Creating timeseries matrix...')
            self.__term_counts_data = self.__tfidf_reduce_obj.create_terms_count(self.__dates)
        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)
        self.__term_score_tuples = utils.stop_tup(self.__term_score_tuples, WordAnalyzer.stemmed_stop_word_set_uni, WordAnalyzer.stemmed_stop_word_set_n)

        # todo: no output method; just if statements to call output functions...?
        #  Only supply what they each directly require

        # todo: hence Pipeline then becomes a single function

    @property
    def term_counts_data(self):
        return self.__term_counts_data

    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50, n_nmf_topics=0):

        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, wordcloud_title=wordcloud_title,
                                  tfidf_reduce_obj=self.__tfidf_reduce_obj, name=outname,
                                  nterms=nterms, term_counts_data=self.__term_counts_data,
                                  date_dict=self.__date_dict, pick=self.__pick_method,
                                  doc_pickle_file_name=self.__data_filename, time=self.__time, nmf_topics=n_nmf_topics)

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

            _, quarterly_values = scripts.utils.date_utils.timeseries_weekly_to_quarterly(weekly_iso_dates, row_values)
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
