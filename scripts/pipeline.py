import bz2
import pickle
from os import makedirs, path

from pandas import read_pickle

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


def remove_empty_documents(data_frame, text_header):
    num_docs_before_sift = data_frame.shape[0]
    data_frame.dropna(subset=[text_header], inplace=True)
    num_docs_after_sift = data_frame.shape[0]
    num_docs_sifted = num_docs_before_sift - num_docs_after_sift
    print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3),
                 normalize_rows=False, text_header='abstract', term_counts=False,
                 pickled_tf_idf_file_name=None, max_df=0.1, user_ngrams=None, tfidf_output=False,
                 output_name=None):

        # load data
        self.__data_filename = data_filename
        self.__date_dict = docs_mask_dict['date']
        self.__time = docs_mask_dict['time']

        self.__pick_method = pick_method
        # calculate or fetch tf-idf mat
        if pickled_tf_idf_file_name is None:

            self.__dataframe = datafactory.get(data_filename)

            remove_empty_documents(self.__dataframe, text_header)

            self.__tfidf_obj = TFIDF(text_series=self.__dataframe[text_header], ngram_range=ngram_range,
                                     max_document_frequency=max_df, tokenizer=LemmaTokenizer())

            self.__text_lengths = self.__dataframe[text_header].map(len).tolist()

            self.__dataframe.drop(columns=[text_header], inplace=True)

            if tfidf_output:
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
        filter_terms_obj = FilterTerms(self.__tfidf_obj.feature_names, user_ngrams,
                                       file_name=path.join('data', 'embeddings', 'glove', 'w2v_glove.6B.50d.txt'))
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
        if term_counts:
            self.__term_counts_data = self.__tfidf_reduce_obj.create_terms_count(self.__dataframe,
                                                                                 docs_mask_dict['date_header'])
        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)

        # todo: no output method; just if statements to call output functions...?
        #  Only supply what they each directly require

        # todo: hence Pipeline then becomes a single function

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
