import calendar
import datetime

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

    for current_row_index in tqdm(range(number_of_rows), 'Calculating weekly term document-counts', unit='document',
                                  total=number_of_rows):
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


def convert_year_week_to_year_week_tuple(date_in_year_week_format):
    year = date_in_year_week_format // 100
    week = (date_in_year_week_format % 100) - 1
    return year, week


def year_week_to_gregorian(date_in_year_week_format):
    "Gregorian calendar date for the given ISO year, week and day"
    # ref: https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar
    iso_year, iso_week = convert_year_week_to_year_week_tuple(date_in_year_week_format)
    iso_day = 1
    fourth_jan = datetime.date(iso_year, 1, 4)
    _, fourth_jan_week, fourth_jan_day = fourth_jan.isocalendar()
    return fourth_jan + datetime.timedelta(days=iso_day - fourth_jan_day, weeks=iso_week - fourth_jan_week)


def add_weeks(year, weeks, offset, num_weeks_per_year):
    weeks += offset
    final_weeks = (weeks % num_weeks_per_year)
    final_years = year + (weeks // num_weeks_per_year)
    return final_years, final_weeks


def convert_year_week_to_fractional_year(date_in_year_week_format, num_weeks_per_year=53):
    year, week = convert_year_week_to_year_week_tuple(date_in_year_week_format)
    return year + (week / num_weeks_per_year)


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


def timeseries_weekly_to_yearly(weekly_dates, weekly_values):
    dict_dates = {}
    for date, value in zip(weekly_dates, weekly_values):
        year = date // 100
        new_date = year * 100

        if new_date in dict_dates:
            dict_dates[new_date] += value
        else:
            dict_dates[new_date] = value

    return list(dict_dates.keys()), list(dict_dates.values())


def date_to_year_week(date):
    iso_date = date.isocalendar()
    integer_date = iso_date[0] * 100 + iso_date[1]
    return integer_date


def generate_year_week_dates(data_frame, date_header):
    if date_header not in data_frame.columns:
        return None

    dates = data_frame[date_header].tolist()
    document_week_dates = [date_to_year_week(d) for d in dates]
    return np.array(document_week_dates, dtype=np.uint32)
