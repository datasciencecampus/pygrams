import array as arr
import datetime
import numpy as np

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from pandas import to_datetime
from pandas.api.types import is_string_dtype


def remove_all_null_rows_global(sparse_mat, df):
    nonzero_row_indices, _ = sparse_mat.nonzero()
    unique_nonzero_indices = np.unique(nonzero_row_indices)

    df = df.reset_index(drop=True)
    df = df.ix[unique_nonzero_indices]
    df = df.reset_index(drop=True)
    return sparse_mat[unique_nonzero_indices], df


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


def convert_year_week_to_year_week_tuple(date_in_year_week_format):
    year = date_in_year_week_format // 100
    week = (date_in_year_week_format % 100) - 1
    return year, week


def iso_to_gregorian(date_in_year_week_format):
    "Gregorian calendar date for the given ISO year, week and day"
    # ref: https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar
    iso_year, iso_week = convert_year_week_to_year_week_tuple(date_in_year_week_format)
    iso_day = 1
    fourth_jan = datetime.date(iso_year, 1, 4)
    _, fourth_jan_week, fourth_jan_day = fourth_jan.isocalendar()
    return fourth_jan + datetime.timedelta(days=iso_day - fourth_jan_day, weeks=iso_week - fourth_jan_week)


def add_weeks( year, weeks, offset, NUM_WEEKS_PER_YEAR):
    weeks += offset
    final_weeks = (weeks % NUM_WEEKS_PER_YEAR)
    final_years = year + (weeks // NUM_WEEKS_PER_YEAR)
    return final_years, final_weeks


def convert_year_week_to_fractional_year( date_in_year_week_format, NUM_WEEKS_PER_YEAR):
    year, week = convert_year_week_to_year_week_tuple(date_in_year_week_format)
    return year + (week / NUM_WEEKS_PER_YEAR)


def get_row_indices_and_values(term_counts_matrix_csc, term_index):
    start_index = term_counts_matrix_csc.indptr[term_index]
    end_index = term_counts_matrix_csc.indptr[term_index + 1]

    return arr.array('i', (term_counts_matrix_csc.indices[start_index:end_index])), \
           arr.array('i', (term_counts_matrix_csc.data[start_index:end_index]))


def timeseries_weekly_to_quarterly(weekly_dates, weekly_values):
    dict_dates = {}
    for date, value in zip(weekly_dates, weekly_values):
        year = date // 100
        week = date % 100
        if 0 <= week < 13:
            new_date = (year * 100) + 1
        elif 13 <= week < 26:
            new_date = (year * 100) + 4
        elif 26 <= week < 39:
            new_date = (year * 100) + 7
        else:
            new_date = (year * 100) + 10

        if new_date in dict_dates:
            dict_dates[new_date] += value
        else:
            dict_dates[new_date] = value

    return list(dict_dates.keys()), list(dict_dates.values())


def fsigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k*(x-x0)))


def fsigmoid_derivative(x, x0, k):
    expon = np.exp(-k*(x-x0))
    return k*expon/((1.0+expon)*(1.0+expon))


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
    new_tokens=[]
    for token in tokensin:
        ngram = token.split()
        if len(ngram)==1:
            if ngram[0] not in unigrams and not ngram[0].isdigit():
                new_tokens.append(token)
        else:
            word_in_ngrams=False
            for word in ngram:
                if word in ngrams or (digits and word.isdigit()):
                    word_in_ngrams=True
                    break
            if not word_in_ngrams:
                new_tokens.append(token)
    return new_tokens


def stop_tup(tuples, unigrams, ngrams, digits=True):
    new_tuples=[]
    for tuple in tuples:
        token = tuple[1]
        ngram = token.split()
        if len(ngram)==1:
            if ngram[0] not in unigrams and not ngram[0].isdigit():
                new_tuples.append(tuple)
        else:
            word_in_ngrams=False
            for word in ngram:
                if word in ngrams or (digits and word.isdigit()):
                    word_in_ngrams=True
                    break
            if not word_in_ngrams:
                new_tuples.append(tuple)
    return new_tuples


def checkdf(df, emtec, docs_mask_dict, text_header, term_counts):
    app_exit = False

    if emtec or docs_mask_dict['time'] or docs_mask_dict['date'] is not None or term_counts:
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