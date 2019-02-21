import numpy as np
from tqdm import tqdm


class TfidfReduce(object):
    def __init__(self, tfidf_mat, feature_names, tfidf_mask, docs_subset, pick_method=sum):
        print("tfidf reduction")
        self.__term_score_tuples = self.__detect_popular_ngrams_in_docs_set( tfidf_mat,feature_names, docs_subset, tfidf_mask, pick_method)

    @property
    def term_score_tuples(self):
        return self.__term_score_tuples

    def __detect_popular_ngrams_in_docs_set(self, tfidf_matrix, feature_names, docs_subset, tfidf_mask, pick_method, verbose=True):
        if docs_subset is None or len(docs_subset)==0:
            print(f'Processing TFIDF of {tfidf_matrix.shape[0]:,} documents')

        if tfidf_matrix.shape[0] == 0:
            print('...skipping as 0 documents...')
            return []

        # pick filter
        tfidf_csc_matrix = tfidf_matrix.tocsc()

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
        feature_iterator = feature_names
        if verbose:
            feature_iterator = tqdm(feature_iterator, leave=False, desc='Searching TFIDF', unit='ngram')

        for ngram_index, ngram in enumerate(feature_iterator):

            start_idx_inptr = tfidf_csc_matrix.indptr[ngram_index]
            end_idx_inptr = tfidf_csc_matrix.indptr[ngram_index + 1]

            non_zero_values_term = tfidf_csc_matrix.data[start_idx_inptr:end_idx_inptr]
            non_zero_values_mask = tfidf_mask[start_idx_inptr:end_idx_inptr]
            non_zero_values_term = [x*y for x,y in zip(non_zero_values_term, non_zero_values_mask)]

            if docs_subset is not None and len(docs_subset)>0:

                row_indices_term = tfidf_csc_matrix.indices[start_idx_inptr:end_idx_inptr]
                non_zero_values_term_set = []

                indices_idx = 0
                for doc_idx in docs_subset:
                    while indices_idx < len(row_indices_term) and row_indices_term[indices_idx] <= doc_idx:
                        if row_indices_term[indices_idx] == doc_idx:
                            non_zero_values_term_set.append(non_zero_values_term[indices_idx] )
                        indices_idx += 1
                non_zero_values_term = non_zero_values_term_set

            if len(non_zero_values_term) > 0:
                pick_value = pick_func(non_zero_values_term)

                if np.isnan(pick_value):
                    pick_value = 0

                ngrams_scores_tuple.append((pick_value, ngram))

        ngrams_scores_tuple.sort(key=lambda tup: -tup[0])
        return ngrams_scores_tuple

