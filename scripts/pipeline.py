import os

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
from scripts.utils.date_utils import year2pandas_earliest_date, year2pandas_latest_date


class Pipeline(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3),
                 normalize_rows=False, text_header='abstract', term_counts=False,
                 pickled_tf_idf=None, max_df=0.1, user_ngrams=None):

        # load data
        self.__data_filename = data_filename

        self.__date_dict = docs_mask_dict['date']
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
                                       docs_mask_dict['date_header'],
                                       text_header=text_header,
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
            self.__term_counts_data = self.__tfidf_reduce_obj.create_terms_count(df, docs_mask_dict['date_header'])
        # if other outputs
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)

    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50):

        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, wordcloud_title=wordcloud_title,
                                  tfidf_reduce_obj=self.__tfidf_reduce_obj, name=outname,
                                  nterms=nterms, term_counts_data=self.__term_counts_data,
                                  tfidf_obj=self.__tfidf_obj, date_dict=self.__date_dict, pick=self.__pick_method,
                                  doc_pickle_file_name=self.__data_filename, time=self.__time)

    @property
    def term_score_tuples(self):
        return self.__term_score_tuples
