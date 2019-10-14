import array as arr
import matplotlib.pyplot as plt
import numpy as np

from bz2 import BZ2File
from os import path, makedirs
from pickle import dump

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from pandas import to_datetime, read_pickle
from pandas.api.types import is_string_dtype


def plot_ngram_bars(ngrams1, ngrams2, dir_name):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)
    labels = ['uni-grams', 'bi-grams', 'tri-grams']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, ngrams1, width, label='original')
    rects2 = ax.bar(x + width / 2, ngrams2, width, label='processed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('counts')
    ax.set_title('ngram counts original vs processed tf-idf matrix')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig(path.join(dir_name, 'ngram_counts.png'))
    plt.close()

def ngrams_count_tups(features_tups):
    ones = sum([1 for x in features_tups if len(x[1].split()) == 1 and x[0] > 0])
    twos = sum([1 for x in features_tups if len(x[1].split()) == 2 and x[0] > 0])
    threes = sum([1 for x in features_tups if len(x[1].split()) == 3 and x[0] > 0])
    return [ones, twos, threes]

def ngrams_counts(features):
    ones = sum([1 for x in features if len(x.split()) == 1])
    twos = sum([1 for x in features if len(x.split()) == 2])
    threes = sum([1 for x in features if len(x.split()) == 3])
    return [ones, twos, threes]


def tfidf_plot(tfidf_obj, message, dir_name=None):
    count_mat = tfidf_obj.count_matrix
    idf = tfidf_obj.idf
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)

    return tfidf_plot2(count_mat, idf, message, dir_name)


def tfidf_plot2(count_mat, idf, message, dir_name):
    counts_arr_sorted = count_mat.toarray().sum(axis=0)
    plt.scatter(counts_arr_sorted, idf, s=5)
    plt.xlabel('sum_tf')
    plt.ylabel('idf')
    plt.title(r'sum_tf vs idf | ' + message)
    plt.savefig(path.join(dir_name, '_'.join(message.split()) + '.png'))
    plt.close()


def histogram(count_matrix):
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    counts_arr_sorted = count_matrix.toarray().sum(axis=0)
    num_bins = 100
    mu = np.mean(counts_arr_sorted)  # mean of distribution
    sigma = np.std(counts_arr_sorted)  # standard deviation of distribution
    # the histogram of the data
    n, bins, patches = plt.hist(counts_arr_sorted, num_bins, facecolor='blue', alpha=0.5)
    # add df vs tf plots here
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.yscale('log')
    plt.xlabel('Sum of term counts')
    plt.ylabel('Num Terms')
    plt.title(r'Histogram of sum of term frequencies')
    print(n)
    print(bins)
    print(patches)
    plt.ylim(bottom=1)
    plt.show()
    exit(0)


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
    if emtec or docs_mask_dict['date'] is not None:
        if docs_mask_dict['date_header'] not in df.columns:
            raise ValueError(f"date_header '{docs_mask_dict['date_header']}' not in dataframe")

    if docs_mask_dict['date_header'] is not None:
        if is_string_dtype(df[docs_mask_dict['date_header']]):
            df[docs_mask_dict['date_header']] = to_datetime(df[docs_mask_dict['date_header']])

            min_date = min(df[docs_mask_dict['date_header']])
            max_date = max(df[docs_mask_dict['date_header']])
            print(f'Document dates range from {min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}')
    else:
        print('Document dates not specified')

    if text_header not in df.columns:
        raise ValueError(f"text_header '{text_header}' not in dataframe")


def remove_empty_documents(data_frame, text_header):
    num_docs_before_sift = data_frame.shape[0]
    data_frame.dropna(subset=[text_header], inplace=True)
    num_docs_after_sift = data_frame.shape[0]
    num_docs_sifted = num_docs_before_sift - num_docs_after_sift
    print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')
