import numpy as np


class TfidfMask(object):
    def __init__(self, tfidf_obj, doc_weights,  unigrams=True,
                 norm_rows=False, max_ngram_length=3, uni_factor=0.8):
        print('creating the tf-idf mask')
        self.__tfidf_matrix = tfidf_obj.tfidf_matrix
        self.__feature_names = tfidf_obj.feature_names
        self.__doc_weights = doc_weights
        self.__text_abstracts = tfidf_obj.text
        self.__tfidf_mask = self.__tfidf_matrix.copy()
        self.__tfidf_mask.data = np.array([1.0 for x in self.__tfidf_matrix.data if x>0.0])
        self.__vectorizer = tfidf_obj.vectorizer
        self.__tf_mat = tfidf_obj.ngram_counts
        self.__uni_factor = uni_factor

        # self.__ngram_counts = csr_matrix(self.__ngram_counts, dtype=np.float64, copy=True)

        # do unigrams
        if unigrams:
            self.__clean_unigrams(self.__max_bigram())

        # normalize rows to text length
        if norm_rows:
            self.__normalize_rows()

        self.__unbias_ngrams(max_ngram_length)

    @property
    def tfidf_mask(self):
        return self.__tfidf_mask

    def update_mask(self, row_weights, column_weights):
        # iterate through rows ( docs)
        for i in range(len(self.__text_abstracts)):
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
        for i in range(len(self.__text_abstracts)):
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
        for i in range(len(self.__text_abstracts)):
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

    def __normalize_rows(self):
        for idx, text in enumerate(self.__text_abstracts):
            text_len = len(text)
            self.__doc_weights /= text_len

    def __unbias_ngrams(self, max_ngram_length):
        # iterate through rows ( docs)
        for i in range(len(self.__text_abstracts)):
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

                    idx_ngram_minus_front = self.__vectorizer.vocabulary_.get(ngram_minus_front)
                    idx_ngram_minus_back = self.__vectorizer.vocabulary_.get(ngram_minus_back)

                    indices_slice = self.__tfidf_matrix.indices[start_idx_ptr:end_idx_ptr]
                    ngram_counts = self.__tfidf_matrix.data[j]

                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_front, ngram_counts, start_idx_ptr)
                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_back, ngram_counts, start_idx_ptr)

    def __unbias_ngrams_slice(self, indices_slice, idx_small_ngram, big_ngram_counts, start_idx_ptr):
        if idx_small_ngram in indices_slice:
            idx = indices_slice.tolist().index(idx_small_ngram)
            small_term_counts = self.__tfidf_matrix.data[start_idx_ptr + idx]
            ratio = 0.0
            if small_term_counts > big_ngram_counts:
                ratio = (small_term_counts - big_ngram_counts) / small_term_counts
            self.__tfidf_mask.data[start_idx_ptr + idx] *= ratio

