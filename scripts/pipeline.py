import pickle
import scripts.data_factory as datafactory
import scripts.output_factory as output_factory
from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.filter_output_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import TFIDF


class Pipeline(object):
    def __init__(self, data_filename, filter_columns, pick_method='sum', max_n=3, min_n=1,
                 normalize_rows=False, text_header='abstract', term_counts=False, dates_header=None,
                 pickled_tf_idf=False, filter_by='union', time=False, citation_dict=None, nterms=25, max_df=0.1,
                 tfidf_wrapper_filename=None):
        # load data
        df = datafactory.get(data_filename)

        # read or create & save tfidf wrapper object
        if pickled_tf_idf and tfidf_wrapper_filename is not None:
            print('reading tfidf wrapper pickle')
            with open(tfidf_wrapper_filename, "rb") as file:
                self.__tfidf_obj = pickle.load(file)
        else:
            self.__tfidf_obj = TFIDF(docs_df=df, ngram_range=(min_n, max_n), max_document_frequency=max_df,
                                     tokenizer=LemmaTokenizer(), text_header=text_header)
            with open(tfidf_wrapper_filename, "wb") as file:
                pickle.dump(self.__tfidf_obj, file)

        # docs subset
        doc_ids = DocumentsFilter(df, filter_columns, filter_by, cpc).doc_indices

        # doc weights
        doc_weights = DocumentsWeights(time, citation_dict).weights

        # tfidf mask ( doc_ids, doc_weights, embeddings_filter will all merge to a single mask in the future)
        tfidf_mask = TfidfMask(self.__tfidf_obj, doc_weights,
                               norm_rows=normalize_rows, max_ngram_length=max_n).tfidf_mask

        self.__tfidf_reduce_obj = TfidfReduce(self.__tfidf_obj, tfidf_mask)
        self.__term_counts_mat = None
        if term_counts:
            self.__term_counts_mat = self.__tfidf_reduce_obj.create_terms_count(df, dates_header)
        # if other outputs
        term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docs_set(doc_ids, pick_method)
        filter_output_obj = FilterTerms(term_score_tuples, nterms=nterms)
        self.__term_score_tuples = filter_output_obj.term_score_tuples

    def output(self, output_types, wordcloud_title=None, outname=None, nterms=50):
        for output_type in output_types:
            output_factory.create(output_type, self.__term_score_tuples, wordcloud_title=wordcloud_title,
                                  tfidf_reduce_obj=self.__tfidf_reduce_obj, name=outname,
                                  nterms=nterms, term_counts_mat=self.__term_counts_mat)
