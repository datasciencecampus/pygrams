import array as arr
from bz2 import BZ2File
from os import path, makedirs
from pickle import dump

import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pandas import to_datetime, read_pickle
from pandas.api.types import is_string_dtype


def fill_missing_zeros(quarterly_values, non_zero_dates, all_quarters):
    for idx, period in enumerate(all_quarters):
        if idx >= len(non_zero_dates):
            non_zero_dates.append(period)
            quarterly_values.append(0)
        elif period == non_zero_dates[idx]:
            continue
        else:
            non_zero_dates.insert(idx, period)
            quarterly_values.insert(idx, 0)
    return non_zero_dates, quarterly_values


def pickle_object(short_name, obj, folder_name):
    makedirs(folder_name, exist_ok=True)
    file_name = pickle_name(short_name, folder_name)
    with BZ2File(file_name, 'wb') as pickle_file:
        dump(obj, pickle_file, protocol=4, fix_imports=False)


def unpickle_object(short_name, folder_name):
    file_name = pickle_name(short_name, folder_name)
    return read_pickle(file_name)


def pickle_name(short_name, folder_name):
    return path.join(folder_name, short_name + '.pkl.bz2')


def unpickle_object( short_name, folder_name):
    file_name = pickle_name(short_name, folder_name)
    return read_pickle(file_name)


def pickle_name( short_name, folder_name):
    return path.join(folder_name, short_name + '.pkl.bz2')


def stationary_terms(emergence_list, nterms):
    if len(emergence_list) == 0:
        return []
    zero_pivot_emergence = 1
    last_emergence = emergence_list[0][1]
    for index, value in enumerate(emergence_list[1:]):
        if value[1] <= 0.0 < last_emergence:
            zero_pivot_emergence = index
            break
        last_emergence = value[1]
    stationary_start_index = zero_pivot_emergence - nterms // 2
    stationary_end_index = zero_pivot_emergence + nterms // 2
    return emergence_list[stationary_start_index:stationary_end_index]


def cpc_dict(df):
    if 'classifications_cpc' not in df.columns:
        return None

    cpc_list_2d = df['classifications_cpc']
    cpc_dict = {}
    for idx, cpc_list in enumerate(cpc_list_2d):
        if not isinstance(cpc_list, list):
            continue
        for cpc_item in cpc_list:
            if cpc_item in cpc_dict:
                cpc_set = cpc_dict[cpc_item]
                cpc_set.add(idx)
                cpc_dict[cpc_item] = cpc_set
            else:
                cpc_dict[cpc_item] = {idx}
    return cpc_dict


def l2normvec(csr_tfidf_mat):
    l2normvec = np.zeros((csr_tfidf_mat.shape[0],), dtype=np.float32)
    # iterate through rows ( docs)
    for i in range(csr_tfidf_mat.shape[0]):
        start_idx_ptr = csr_tfidf_mat.indptr[i]
        end_idx_ptr = csr_tfidf_mat.indptr[i + 1]
        l2norm = 0
        # iterate through columns with non-zero entries
        for j in range(start_idx_ptr, end_idx_ptr):
            tfidf_val = csr_tfidf_mat.data[j]
            l2norm += tfidf_val * tfidf_val
        l2normvec[i] = (np.sqrt(l2norm))
    return l2normvec


def apply_l2normvec(csr_tfidf_mat, l2normvec):
    # iterate through rows ( docs)
    for i in range(csr_tfidf_mat.shape[0]):
        start_idx_ptr = csr_tfidf_mat.indptr[i]
        end_idx_ptr = csr_tfidf_mat.indptr[i + 1]
        l2norm = l2normvec[i]
        # iterate through columns with non-zero entries
        for j in range(start_idx_ptr, end_idx_ptr):
            csr_tfidf_mat.data[j] /= l2norm
    return csr_tfidf_mat


def remove_all_null_rows_global(sparse_mat, dates):
    nonzero_row_indices, _ = sparse_mat.nonzero()
    unique_nonzero_indices = np.unique(nonzero_row_indices)

    return sparse_mat[unique_nonzero_indices], \
           dates[unique_nonzero_indices] if dates is not None else None


def bisearch_csr(array, target, start, end):
    while start <= end:
        middle = (start + end) // 2
        midpoint = array[middle]
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return middle, array[middle] == target
    return 0, False


def remove_all_null_rows(sparse_mat):
    nonzero_row_indices, _ = sparse_mat.nonzero()
    unique_nonzero_indices = np.unique(nonzero_row_indices)
    return sparse_mat[unique_nonzero_indices]


def normalize_array(X, min_val=0.2, return_list=False):
    min_x, max_x = min(X), max(X)
    diff_x = (max_x - min_x)
    std_x = (np.array(X) - min_x) / diff_x
    x_scaled = std_x * (1 - min_val) + min_val
    return x_scaled if not return_list else list(x_scaled)


def w2vify(filein, fileout):
    glove_file = datapath(filein)
    tmp_file_name = get_tmpfile(fileout)
    _ = glove2word2vec(glove_file, tmp_file_name)
    return KeyedVectors.load_word2vec_format(tmp_file_name)


def get_row_indices_and_values(term_counts_matrix_csc, term_index):
    start_index = term_counts_matrix_csc.indptr[term_index]
    end_index = term_counts_matrix_csc.indptr[term_index + 1]

    return arr.array('i', (term_counts_matrix_csc.indices[start_index:end_index])), \
           arr.array('i', (term_counts_matrix_csc.data[start_index:end_index]))


def fsigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fsigmoid_derivative(x, x0, k):
    expon = np.exp(-k * (x - x0))
    return k * expon / ((1.0 + expon) * (1.0 + expon))


def fit_score(y, y_fit):
    # residual sum of squares
    y = np.asarray(y)
    y_fit = np.asarray(y_fit)
    ss_res = np.sum((y - y_fit) ** 2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)
    return r2


def normalize(ydata):
    miny = min(ydata)
    maxy = max(ydata)
    diff = (maxy - miny)

    return np.asarray([(_y - miny) / diff for _y in ydata])


def stop(tokensin, unigrams, ngrams, digits=True):
    new_tokens = []
    for token in tokensin:
        ngram = token.split()
        if len(ngram) == 1:
            if ngram[0] not in unigrams and not ngram[0].isdigit():
                new_tokens.append(token)
        else:
            word_in_ngrams = False
            for word in ngram:
                if word in ngrams or (digits and word.isdigit()):
                    word_in_ngrams = True
                    break
            if not word_in_ngrams:
                new_tokens.append(token)
    return new_tokens


def stop_tup(tuples, unigrams, ngrams, digits=True):
    new_tuples = []
    for tuple in tuples:
        token = tuple[1]
        ngram = token.split()
        if len(ngram) == 1:
            if ngram[0] not in unigrams and not ngram[0].isdigit():
                new_tuples.append(tuple)
        else:
            word_in_ngrams = False
            for word in ngram:
                if word in ngrams or (digits and word.isdigit()):
                    word_in_ngrams = True
                    break
            if not word_in_ngrams:
                new_tuples.append(tuple)
    return new_tuples


def checkdf(df, emtec, docs_mask_dict, text_header):
    app_exit = False

    if emtec or docs_mask_dict['date'] is not None:
        if docs_mask_dict['date_header'] not in df.columns:
            print(f"date_header '{docs_mask_dict['date_header']}' not in dataframe")
            app_exit = True

    if docs_mask_dict['date_header'] is not None:
        if is_string_dtype(df[docs_mask_dict['date_header']]):
            df[docs_mask_dict['date_header']] = to_datetime(df[docs_mask_dict['date_header']])

            min_date = min(df[docs_mask_dict['date_header']])
            max_date = max(df[docs_mask_dict['date_header']])
            print(f'Document dates range from {min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}')
    else:
        print('Document dates not specified')

    if text_header not in df.columns:
        print(f"text_header '{text_header}' not in dataframe")
        app_exit = True

    if app_exit:
        exit(0)


def remove_empty_documents(data_frame, text_header):
    num_docs_before_sift = data_frame.shape[0]
    data_frame.dropna(subset=[text_header], inplace=True)
    num_docs_after_sift = data_frame.shape[0]
    num_docs_sifted = num_docs_before_sift - num_docs_after_sift
    print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')
