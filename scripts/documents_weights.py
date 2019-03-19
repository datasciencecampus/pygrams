import datetime

import numpy as np
from tqdm import tqdm

import scripts.utils.utils as ut


class DocumentsWeights(object):
    def __init__(self, df, time, citation_count_dict, date_header, text_lengths=None, norm_rows=False):
        self.__dataframe = df
        self.__date_header = date_header
        self.__weights = [1.0]*len(df)
        processed = False
        if time:
            time_weights = self.__time_weights()
            self.__weights = [a * b for a, b in zip(self.__weights, time_weights)]
            processed=True

        if citation_count_dict:
            cite_weights = self.__citation_weights(citation_count_dict)
            self.__weights = [a * b for a, b in zip(self.__weights, cite_weights)]
            processed = True

        # normalize rows to text length
        if norm_rows:
            self.__normalize_rows(text_lengths)
            processed = True

        if processed:
            self.__weights = ut.normalize_array(self.__weights, return_list=True)

    @property
    def weights(self):
        return self.__weights

    def __normalize_rows(self, text_lengths):
        self.__weights = [a / b for a, b in zip(self.__weights, text_lengths)]

    def __time_weights(self):
        self.__dataframe = self.__dataframe.sort_values(by=self.__date_header)
        epoch = datetime.datetime.utcfromtimestamp(0)
        X = [(date - epoch).total_seconds() for date in self.__dataframe[self.__date_header]]
        min_x, max_x = min(X), max(X)
        return (np.array(X) - min_x) / (max_x-min_x)

    def __citation_weights(self, citation_count_dict):
        # TODO check if we need -2 below. If not, we only need one dict for both citations and docs_set
        doc_id_dict = {k[:-2]: v for v, k in enumerate(self.__dataframe[self.__id_header])}
        citation_count_for_doc_id_dict = {}
        for key, _ in tqdm(doc_id_dict.items()):
            citation_count_for_doc_id_dict[key] = citation_count_dict.get(key, .0)

        max_citation_count_val = float(max(citation_count_for_doc_id_dict.values()))
        min_citation_count_val = 0.05

        if max_citation_count_val == 0:
            for doc_id in citation_count_for_doc_id_dict:
                citation_count_for_doc_id_dict[doc_id] = 1.0
        else:
            for doc_id in citation_count_for_doc_id_dict:
                citation_count_for_doc_id_dict_std = min_citation_count_val + (
                        (float(citation_count_for_doc_id_dict[doc_id]) - min_citation_count_val) / (
                         max_citation_count_val - min_citation_count_val))
                citation_count_for_doc_id_dict[doc_id] = citation_count_for_doc_id_dict_std

        return list(citation_count_for_doc_id_dict.values())

        # for i, v in enumerate(list_of_citation_counts):
        #     self.__tfidf_matrix.data[self.__tfidf_matrix.indptr[i]:self.__tfidf_matrix.indptr[i + 1]] *= v