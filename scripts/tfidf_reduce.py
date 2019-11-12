from math import log
import numpy as np
from tqdm import tqdm

from scripts.utils.date_utils import tfidf_with_dates_to_weekly_term_counts


class TfidfReduce(object):
    def __init__(self, tfidf_masked, feature_names, tfidf_obj=None, lens=None):
        self.__tfidf_masked = tfidf_masked
        self.__feature_names = feature_names
        self.__lens = lens
        if tfidf_obj is not None:
            self.__tfidf_mat = tfidf_obj.tfidf_matrix
            self.__tfidf_mat_normalized = self.__normalized_count_matrix(lens, self.__tfidf_mat.copy())

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

    def extract_ngrams_from_docset(self, pick_method, verbose=True,  normalised=False):

        if self.__tfidf_masked.getformat() == 'csr':
            self.__tfidf_masked = self.__tfidf_masked.tocsc()

        N = self.__tfidf_masked.shape[0]

        ngrams_scores_tuple = []
        feature_iterator = self.__feature_names
        if verbose:
            feature_iterator = tqdm(feature_iterator, leave=False, desc='Searching TFIDF', unit='ngram')

        for ngram_index, ngram in enumerate(feature_iterator):
            start_idx_inptr = self.__tfidf_masked.indptr[ngram_index]
            end_idx_inptr = self.__tfidf_masked.indptr[ngram_index + 1]

            non_zero_values_term = self.__tfidf_masked.data[start_idx_inptr:end_idx_inptr]
            if len(non_zero_values_term) > 0:
                if pick_method == 'median':
                    pick_value = np.median(non_zero_values_term)
                elif pick_method == 'avg':
                    pick_value = np.average(non_zero_values_term)
                elif pick_method == 'max':
                    pick_value = np.max(non_zero_values_term)
                elif pick_method == 'sum':
                    pick_value = np.sum(non_zero_values_term)
                elif pick_method == 'mean_prob':
                    pick_value = np.sum(non_zero_values_term)/N
                elif pick_method == 'entropy':
                    pick_value = np.sum([pij * log(1 / pij) for pij in non_zero_values_term])
                elif pick_method == 'variance':
                    mpj = np.sum(non_zero_values_term)/N
                    pick_value = np.sum([(pij - mpj) ** 2 for pij in non_zero_values_term])

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
