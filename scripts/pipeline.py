from scripts.data_factory import DataFactory
from scripts.documents_filter import DocumentsFilter
from scripts.tfidf_wraper import TFIDF
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.output_factory import OutputFactory


class Pipeline(object):
    def __init_(self, pickled_tf_idf, filters, pick_method):
        print()
        if pickled_tf_idf():
            print('read from pickle')
        else:
            df = DataFactory.get_data_frame()
            filter = DocumentsFilter(df, filters)
            doc_ids=filter.get_document_indices()
            tfidf_obj = TFIDF(docs_df=df)
            tfidf_mat = tfidf_obj.tfidf_matrix
            tfidf_mask_object = TfidfMask(tfidf_mat)
            tfidf_mask = tfidf_mask_object.get_mask()
            tfidf_reduce_obj = TfidfReduce(tfidf_mask, doc_ids, pick_method)
            self.__term_score_tuples = tfidf_reduce_obj.get_term_score_tuples()

    def output(self, output_type):
        return OutputFactory.get(output_type)
