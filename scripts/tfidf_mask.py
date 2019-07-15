import numpy as np


class TfidfMask(object):
    def __init__(self, tfidf_obj, ngram_range=(2, 3), uni_factor=0.8, unbias=False):
        self.__tfidf_matrix = tfidf_obj.tfidf_matrix
        self.__feature_names = tfidf_obj.feature_names
        self.__tfidf_mask = self.__tfidf_matrix.copy()
        self.__tfidf_mask.data = np.ones(len(self.__tfidf_matrix.data))
        self.__vocabulary = tfidf_obj.vocabulary
        self.__uni_factor = uni_factor
        self.__idf = tfidf_obj.idf

        if unbias:
            # do unigrams
            if ngram_range[0] == 1:
                self.__clean_unigrams(self.__max_bigram())

            for i in range(ngram_range[0], ngram_range[1]):
                self.__unbias_ngrams(i + 1)

    @property
    def tfidf_mask(self):
        return self.__tfidf_mask

    def update_mask(self, row_weights, column_weights):
        # iterate through rows ( docs)
        for i in range(self.__tfidf_matrix.shape[0]):
            start_idx_ptr = self.__tfidf_matrix.indptr[i]
            end_idx_ptr = self.__tfidf_matrix.indptr[i + 1]
            row_weight = row_weights[i]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):
                col_idx = self.__tfidf_matrix.indices[j]
                col_weight = column_weights[col_idx]

                self.__tfidf_mask.data[j] *= (col_weight * row_weight)

    def __clean_unigrams(self, max_bi_freq):
        # iterate through rows ( docs)
        for i in range(self.__tfidf_matrix.shape[0]):
            start_idx_ptr = self.__tfidf_matrix.indptr[i]
            end_idx_ptr = self.__tfidf_matrix.indptr[i + 1]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):

                col_idx = self.__tfidf_matrix.indices[j]
                ngram = self.__feature_names[col_idx]
                ngram_terms = ngram.split()

                if len(ngram_terms) == 1:
                    if self.__tfidf_matrix.data[j] < self.__uni_factor * max_bi_freq:
                        self.__tfidf_mask.data[j] = 0.0
        return 0

    def __max_bigram(self):
        max_tf = 0.0
        # iterate through rows ( docs)
        for i in range(self.__tfidf_matrix.shape[0]):
            start_idx_ptr = self.__tfidf_matrix.indptr[i]
            end_idx_ptr = self.__tfidf_matrix.indptr[i + 1]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):

                col_idx = self.__tfidf_matrix.indices[j]
                ngram = self.__feature_names[col_idx]
                ngram_terms = ngram.split()

                if len(ngram_terms) == 2:
                    max_tf = max(self.__tfidf_matrix.data[j], max_tf)
        return max_tf

    def __unbias_ngrams(self, max_ngram_length):
        # iterate through rows ( docs)
        for i in range(self.__tfidf_matrix.shape[0]):
            start_idx_ptr = self.__tfidf_matrix.indptr[i]
            end_idx_ptr = self.__tfidf_matrix.indptr[i + 1]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):
                col_idx = self.__tfidf_matrix.indices[j]
                big_ngram = self.__feature_names[col_idx]
                big_ngram_terms = big_ngram.split()

                if len(big_ngram_terms) == max_ngram_length:
                    ngram_minus_front = ' '.join(big_ngram_terms[1:])
                    ngram_minus_back = ' '.join(big_ngram_terms[:len(big_ngram_terms) - 1])

                    idx_ngram_minus_front = self.__vocabulary.get(ngram_minus_front)
                    idx_ngram_minus_back = self.__vocabulary.get(ngram_minus_back)

                    indices_slice = self.__tfidf_matrix.indices[start_idx_ptr:end_idx_ptr]
                    ngram_counts = self.__tfidf_matrix.data[j] / self.__idf[col_idx]

                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_front, ngram_counts, start_idx_ptr)
                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_back, ngram_counts, start_idx_ptr)

    def __unbias_ngrams_slice(self, indices_slice, idx_small_ngram, big_ngram_counts, start_idx_ptr):
        if idx_small_ngram in indices_slice:
            idx = indices_slice.tolist().index(idx_small_ngram)
            small_term_counts = self.__tfidf_matrix.data[start_idx_ptr + idx] / self.__idf[idx_small_ngram]
            ratio = 0.0
            if abs(small_term_counts - big_ngram_counts) > 0.000001:
                ratio = (small_term_counts - big_ngram_counts) / small_term_counts
            self.__tfidf_mask.data[start_idx_ptr + idx] *= ratio
