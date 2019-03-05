import calendar

import numpy as np
from pandas import Timestamp
from scipy.sparse import csr_matrix, vstack, isspmatrix_csr
from tqdm import tqdm


def choose_last_day(year_in, month_in):
    return str(calendar.monthrange(int(year_in), int(month_in))[1])


def year2pandas_latest_date(year_in, month_in):
    if year_in is None:
        return Timestamp.now()

    if month_in is None:
        return Timestamp(str(year_in) + '-12-31')

    year_string = str(year_in) + '-' + str(month_in) + '-' + choose_last_day(year_in, month_in)
    return Timestamp(year_string)


def year2pandas_earliest_date(year_in, month_in):
    if year_in is None:
        return Timestamp('2000-01-01')

    if month_in is None:
        return Timestamp(str(year_in) + '-01-01')

    year_string = str(year_in) + '-' + str(month_in) + '-01'
    return Timestamp(year_string)


def tfidf_with_dates_to_weekly_term_counts(term_value_array, uspto_week_dates):
    number_of_rows, number_of_terms = term_value_array.shape
    week_counts_csr = None

    if not isspmatrix_csr(term_value_array):
        term_value_array = csr_matrix(term_value_array)

    current_week = int(uspto_week_dates[0])
    current_week_counts_csr = csr_matrix((1, number_of_terms), dtype=np.int32)
    week_totals = []
    week_dates = []
    week_total = 0

    for current_row_index in tqdm(range(number_of_rows), 'Counting terms per week', unit='patent'):
        new_week = int(uspto_week_dates[current_row_index])

        while new_week > current_week:
            if ((current_week % 100) == 53) and (week_total == 0):
                current_week += 100 - 53 + 1  # next year, so add 100 but remove the "used" weeks and move on by 1
            else:
                week_counts_csr = vstack([week_counts_csr, current_week_counts_csr],
                                         format='csr') if week_counts_csr is not None else current_week_counts_csr
                week_totals.append(week_total)
                week_dates.append(current_week)
                current_week_counts_csr = csr_matrix((1, number_of_terms), dtype=np.int32)
                current_week += 1
                if (current_week % 100) > 53:
                    current_week += 100 - 53  # next year, so add 100 but remove the "used" weeks
                week_total = 0

        current_row_as_counts = term_value_array[current_row_index, :] > 0
        current_week_counts_csr += current_row_as_counts
        week_total += 1

    week_counts_csr = vstack([week_counts_csr, current_week_counts_csr],
                             format='csr') if week_counts_csr is not None else current_week_counts_csr
    week_totals.append(week_total)
    week_dates.append(current_week)

    return week_counts_csr, week_totals, week_dates
