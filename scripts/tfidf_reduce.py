from math import log, sqrt

import numpy as np
from tqdm import tqdm

from scripts.utils.date_utils import tfidf_with_dates_to_weekly_term_counts


class TfidfReduce(object):
    def __init__(self, tfidf_masked, feature_names, tfidf_obj=None, lens=None):
        self.__tfidf_masked = tfidf_masked
        self.__feature_names = feature_names
        self.__lens = lens
        if tfidf_obj is not None:
            self.__ngram_count = tfidf_obj.count_matrix
            self.__tf_normalized = self.__normalized_count_matrix(lens, self.__ngram_count.copy())

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def tfidf_masked(self):
        return self.__tfidf_masked

    def __normalized_count_matrix(self, lens, csr_mat):
        csr_mat.data = np.array([np.float32(round(x)) for x in csr_mat.data])
        for i in range(csr_mat.shape[0]):
            start_idx_ptr = csr_mat.indptr[i]
            end_idx_ptr = csr_mat.indptr[i + 1]
            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                csr_mat.data[j] /= lens[i]
        return csr_mat

    def extract_ngrams_from_row(self, row_num):
        if self.__tfidf_masked.getformat() == 'csc':
            self.__tfidf_masked = self.__tfidf_masked.tocsr()

        ngrams_scores_tuple = []

        start_idx_ptr = self.__tfidf_masked.indptr[row_num]
        end_idx_ptr = self.__tfidf_masked.indptr[row_num + 1]

        # iterate through columns(ngrams) with non-zero entries
        for j in range(start_idx_ptr, end_idx_ptr):
            col_idx = self.__tfidf_masked.indices[j]
            ngram = self.feature_names[col_idx]

            pick_value = self.__tfidf_masked.data[j]
            if np.isnan(pick_value):
                pick_value = 0.0
            ngrams_scores_tuple.append((pick_value, ngram))
        return ngrams_scores_tuple

    def collect_vector_for_feature(self, csc_mat):
        mp_vec=[]
        for j in range(csc_mat.shape[1]):
            start_idx_ptr = csc_mat.indptr[j]
            end_idx_ptr = csc_mat.indptr[j + 1]
            mpj=0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                pij = csc_mat.data[i]
                mpj+=pij
            mp_vec.append(mpj)
        return np.array(mp_vec)

    def get_zipf_scores(self):
        tf_norm = self.__tf_normalized
        tfidf_score = self.__tfidf_masked
        ngram_count = self.__ngram_count

        tf_norm = tf_norm.tocsc()
        tfidf = tfidf_score.tocsc()
        count = ngram_count.tocsc()

        if not tf_norm.getformat() == 'csc':
            print('problem')

        N=tf_norm.shape[0]
        tot_pij =  self.collect_vector_for_feature(tf_norm)
        mp_vec = tot_pij/N
        tfidf_vec = self.collect_vector_for_feature(tfidf)
        ngram_vec = self.collect_vector_for_feature(count)
        probabilities_vec = mp_vec * N

        variance_vec = []
        for j in range(tf_norm.shape[1]):
            start_idx_ptr = tf_norm.indptr[j]
            end_idx_ptr = tf_norm.indptr[j + 1]
            vpj = 0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                row_idx = tf_norm.indices[i]
                pij = tf_norm.data[i]
                vpj += (pij - mp_vec[j]) ** 2 * self.__lens[row_idx]
            variance = vpj
            variance_vec.append(variance)

        entropy_vec = []
        for j in range(tf_norm.shape[1]):
            start_idx_ptr = tf_norm.indptr[j]
            end_idx_ptr = tf_norm.indptr[j + 1]
            entropy_j=0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                pij = tf_norm.data[i]
                entropy_j += pij * log(1/pij)
            entropy_vec.append(entropy_j)

        sat_vec = mp_vec / np.sqrt(np.array(variance_vec))

        return variance_vec, entropy_vec, sat_vec, tfidf_vec, probabilities_vec, ngram_vec

    def extract_ngrams_from_docset(self, pick_method, verbose=True):
        if self.__tfidf_masked.getformat() == 'csr':
            self.__tfidf_masked = self.__tfidf_masked.tocsc()

        if pick_method == 'median':
            pick_func = np.median
        elif pick_method == 'avg':
            pick_func = np.average
        elif pick_method == 'max':
            pick_func = np.max
        else:
            pick_func = np.sum

        ngrams_scores_tuple = []
        feature_iterator = self.__feature_names
        if verbose:
            feature_iterator = tqdm(feature_iterator, leave=False, desc='Searching TFIDF', unit='ngram')

        for ngram_index, ngram in enumerate(feature_iterator):
            start_idx_inptr = self.__tfidf_masked.indptr[ngram_index]
            end_idx_inptr = self.__tfidf_masked.indptr[ngram_index + 1]

            non_zero_values_term = self.__tfidf_masked.data[start_idx_inptr:end_idx_inptr]
            if len(non_zero_values_term) > 0:
                pick_value = pick_func(non_zero_values_term)

                if np.isnan(pick_value):
                    pick_value = 0

                ngrams_scores_tuple.append((pick_value, ngram))

        ngrams_scores_tuple.sort(key=lambda tup: -tup[0])
        return ngrams_scores_tuple

    def create_timeseries_data(self, document_week_dates):
        term_counts_per_week, number_of_documents_per_week, year_week_dates = tfidf_with_dates_to_weekly_term_counts(
            self.__tfidf_masked, document_week_dates)

        term_counts_data = [term_counts_per_week, self.__feature_names, number_of_documents_per_week,
                            year_week_dates]

        return term_counts_data