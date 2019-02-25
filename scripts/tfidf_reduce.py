import numpy as np
from tqdm import tqdm

from scripts.utils.datesToPeriods import tfidf_with_dates_to_weekly_term_counts


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

    def extract_row_from_mask(self, row_num):
        if self.__tfidf_masked.getformat() == 'csc':
            self.__tfidf_masked = self.__tfidf_masked.tocsr()

        ngrams_scores_tuple = []

        start_idx_ptr = self.__tfidf_masked.indptr[row_num]
        end_idx_ptr = self.__tfidf_masked.indptr[row_num + 1]

        # iterate through columns(ngrams) with non-zero entries
        for j in range(start_idx_ptr, end_idx_ptr):
            col_idx = self.__tfidf_masked.indices[j]
            ngram=self.feature_names[col_idx]
            pick_value = self.__tfidf_masked.data[j]
            if np.isnan(pick_value):
                pick_value = 0
            ngrams_scores_tuple.append((pick_value, ngram))
        return ngrams_scores_tuple

    def extract_nbest_from_mask(self, pick_method, verbose=True):
        if self.__tfidf_masked.getformat() == 'csr':
            self.__tfidf_masked = self.__tfidf_masked.tocsc()

        if pick_method == 'median':
            pick_func = np.median
        elif pick_method == 'avg':
            pick_func = np.average
        elif pick_method == 'max':
            pick_func = np.max
        elif pick_method == 'sum':
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

    # obsolete
    def extract_ngrams_from_docs_set(self, docs_subset, pick_method, verbose=True):
        if docs_subset is None or len(docs_subset)==0:
            print(f'Processing TFIDF of {self.__tfidf_matrix.shape[0]:,} documents')

        if self.__tfidf_matrix.shape[0] == 0:
            print('...skipping as 0 documents...')
            return []

        # pick filter
        tfidf_csc_matrix = self.__tfidf_matrix.tocsc()

        if pick_method == 'median':
            pick_func = np.median
        elif pick_method == 'avg':
            pick_func = np.average
        elif pick_method == 'max':
            pick_func = np.max
        elif pick_method == 'sum':
            pick_func = np.sum

        # for i, v in enumerate(time_weights):
        #     self.__tfidf_matrix.data[self.__tfidf_matrix.indptr[i]:self.__tfidf_matrix.indptr[i + 1]] *= v

        ngrams_scores_tuple = []
        feature_iterator = self.__feature_names
        if verbose:
            feature_iterator = tqdm(feature_iterator, leave=False, desc='Searching TFIDF', unit='ngram')



        for ngram_index, ngram in enumerate(feature_iterator):

            start_idx_inptr = tfidf_csc_matrix.indptr[ngram_index]
            end_idx_inptr = tfidf_csc_matrix.indptr[ngram_index + 1]

            non_zero_values_term = tfidf_csc_matrix.data[start_idx_inptr:end_idx_inptr]
            non_zero_values_mask = self.__tfidf_mask[start_idx_inptr:end_idx_inptr]
            non_zero_values_term = [x*y for x,y in zip(non_zero_values_term, non_zero_values_mask)]

            if docs_subset is not None and len(docs_subset)>0:

                row_indices_term = tfidf_csc_matrix.indices[start_idx_inptr:end_idx_inptr]
                non_zero_values_term_set = []

                indices_idx = 0
                for doc_idx in docs_subset:
                    while indices_idx < len(row_indices_term) and row_indices_term[indices_idx] <= doc_idx:
                        if row_indices_term[indices_idx] == doc_idx:
                            non_zero_values_term_set.append(non_zero_values_term[indices_idx])
                        indices_idx += 1
                non_zero_values_term = non_zero_values_term_set

            if len(non_zero_values_term) > 0:
                pick_value = pick_func(non_zero_values_term)

                if np.isnan(pick_value):
                    pick_value = 0

                ngrams_scores_tuple.append((pick_value, ngram))

        ngrams_scores_tuple.sort(key=lambda tup: -tup[0])
        return ngrams_scores_tuple

    def create_terms_count(self, df, dates_header):
        try:
            dates = df[dates_header]
            document_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                                   [d.isocalendar() for d in dates]]
        except ValueError:
            # do we need this?
            dates = [None] * len(df)

        term_counts_per_week, number_of_documents_per_week, week_iso_dates = tfidf_with_dates_to_weekly_term_counts(
            self.__tfidf_matrix, document_week_dates)

        term_counts_data = [term_counts_per_week, self.__feature_names, number_of_documents_per_week,
                            week_iso_dates]

        return term_counts_data
