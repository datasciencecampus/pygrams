import scripts.data_factory as datafactory
from scripts.documents_filter import DocumentsFilter
from scripts.documents_weights import DocumentsWeights
from scripts.filter_output_terms import FilterTerms
from scripts.text_processing import LemmaTokenizer
from scripts.tfidf_wraper import TFIDF
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
import scripts.output_factory as output_factory


class Pipeline(object):
    def __init__(self, data_filename,  filter_columns, cpc = None, pick_method='sum', max_n=3, min_n=1, normalize_rows=False,
                 pickled_tf_idf=False, filter_by='union', time=False, citation_dict=None, nterms=25, max_df = 0.1):
        print()
        if pickled_tf_idf:
            print('read from pickle')
        else:
            df = datafactory.get(data_filename)
            doc_filter_obj = DocumentsFilter(df, filter_columns, filter_by, cpc)
            doc_ids=doc_filter_obj.doc_indices
            doc_weights_obj = DocumentsWeights(time, citation_dict)

            tfidf_obj = TFIDF(docs_df=df, ngram_range=(min_n, max_n), max_document_frequency=max_df,
                              tokenizer=LemmaTokenizer())
            tfidf_mat = tfidf_obj.tfidf_matrix
            tfidf_mask_object = TfidfMask(tfidf_mat, tfidf_obj.ngram_counts, tfidf_obj.feature_names, doc_weights_obj.weights, tfidf_obj.text,
                                          tfidf_obj.vectorizer, norm_rows=normalize_rows, max_ngram_length=max_n)
            tfidf_mask = tfidf_mask_object.get_mask()
            tfidf_reduce_obj = TfidfReduce(tfidf_mat, tfidf_obj.feature_names, tfidf_mask, doc_ids, pick_method)
            term_score_tuples = tfidf_reduce_obj.term_score_tuples
            filter_output_obj = FilterTerms(term_score_tuples,nterms=nterms)
            self.__term_score_tuples = filter_output_obj.term_score_tuples
            print('done')

    def output(self, output_types):
        for output_type in output_types:
            output_factory.get(output_type, self.__term_score_tuples)
