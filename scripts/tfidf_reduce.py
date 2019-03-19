import numpy as np
from tqdm import tqdm

from scripts.utils.date_utils import tfidf_with_dates_to_weekly_term_counts


class TfidfReduce(object):
    def __init__(self, tfidf_masked, feature_names):
        self.__tfidf_masked = tfidf_masked
        self.__feature_names = feature_names

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def tfidf_masked(self):
        return self.__tfidf_masked

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

    def create_terms_count(self, df, dates_header):
        dates = df[dates_header]
        document_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                               [d.isocalendar() for d in dates]]

        term_counts_per_week, number_of_documents_per_week, week_iso_dates = tfidf_with_dates_to_weekly_term_counts(
            self.__tfidf_masked, document_week_dates)

        term_counts_data = [term_counts_per_week, self.__feature_names, number_of_documents_per_week,
                            week_iso_dates]

        return term_counts_data
