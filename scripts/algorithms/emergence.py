import array as arr

import numpy as np
import pylab as plt

from datetime import date
from math import ceil, sqrt

from scipy.optimize import curve_fit
from scripts.utils.utils import fsigmoid, fsigmoid_derivative, fit_score, normalize


class Emergence(object):

    def __init__(self, number_of_patents_per_week, num_weeks_period=52):
        self.TERM_BASE_RECS_THRESHOLD = 5000
        self.BASE_TERM2ALL_RATIO_THRESHOLD = 0.15
        self.MIN_DOCS_FOR_EMERGENCE = 7

        self.NUM_PERIODS_BASE = 3
        self.NUM_PERIODS_ACTIVE = 7
        self.NUM_PERIODS = self.NUM_PERIODS_BASE + self.NUM_PERIODS_ACTIVE

        self.NUM_WEEKS_PER_PERIOD = num_weeks_period

        self.__number_of_patents_per_week = arr.array('i', number_of_patents_per_week)

        self.__today_year, self.__today_week, _ = date.today().isocalendar()

        self.__per_period_counts_all = arr.array('i', [0] * self.NUM_PERIODS)
        self.__per_period_counts_term = arr.array('i', [0] * self.NUM_PERIODS)

    def __check_records(self, num_term_records, term_start_week, term_end_week, porter=False):
        if num_term_records < self.MIN_DOCS_FOR_EMERGENCE:
            return False

        diff_periods = (term_end_week - term_start_week + 1) / self.NUM_WEEKS_PER_PERIOD

        if ceil(diff_periods) < self.NUM_PERIODS and porter:
            return False

        return True

    def __calculate_counts_term(self, term_weeks, term_counts, start_dates):
        first_term_index_in_period = arr.array('i', [0] * (self.NUM_PERIODS + 1))
        period = 0

        for term_index in range(1, len(term_weeks)):
            while period < self.NUM_PERIODS and term_weeks[term_index] >= start_dates[period + 1]:
                period += 1
                first_term_index_in_period[period] = term_index

            if period == self.NUM_PERIODS:
                break

        if period < self.NUM_PERIODS:
            first_term_index_in_period[self.NUM_PERIODS] = len(term_weeks)

        counts_term = [sum(term_counts[first_term_index_in_period[period]:first_term_index_in_period[period + 1]])
                       for period in range(self.NUM_PERIODS)]

        return counts_term

    def init_vars(self, term_weeks, term_counts, porter=False):
        # TODO: if len(term_counts) > self.NUM_WEEKS_PER_PERIOD has the same accuracy and 40% performance increase?

        num_term_records = len(term_weeks)
        if num_term_records == 0:
            return False

        term_start_week = term_weeks[0]
        term_end_week = term_weeks[-1]

        if self.__check_records(num_term_records, term_start_week, term_end_week, porter=porter):
            start_dates = arr.array('i', [term_start_week + period * self.NUM_WEEKS_PER_PERIOD
                                          for period in range(self.NUM_PERIODS + 1)])
            self.__per_period_counts_term = self.__calculate_counts_term(term_weeks, term_counts, start_dates)

            num_records_in_active_period = sum(self.__per_period_counts_term[self.NUM_PERIODS_BASE:])
            if num_records_in_active_period != 0:
                self.__per_period_counts_all = [
                    sum(self.__number_of_patents_per_week[start_dates[period]:start_dates[period + 1]])
                    for period in range(self.NUM_PERIODS)]
                return True

        return False

    def calculate_emergence(self, term_weeks):
        num_term_records = len(term_weeks)
        term_start_week = term_weeks[0]
        term_end_week = term_weeks[-1]

        num_records_base_term = sum(self.__per_period_counts_term[:self.NUM_PERIODS_BASE])
        num_records_base_all = sum(self.__per_period_counts_all[:self.NUM_PERIODS_BASE])

        base2all_below_threshold = num_records_base_term / num_records_base_all < self.BASE_TERM2ALL_RATIO_THRESHOLD
        appears_records_at_least_3_years = (term_end_week - term_start_week) >= 3
        at_least_n_recs = num_term_records >= self.MIN_DOCS_FOR_EMERGENCE
        active2base_ratio = (num_term_records - num_records_base_term) / num_records_base_term

        return appears_records_at_least_3_years and at_least_n_recs and active2base_ratio > 2 \
            and base2all_below_threshold and self.has_multiple_author_sets()

    @staticmethod
    def has_multiple_author_sets():
        return True

    def calculate_escore(self):
        term_counts = self.__per_period_counts_term[self.NUM_PERIODS_BASE:]
        total_counts = self.__per_period_counts_all[self.NUM_PERIODS_BASE:]

        sum_sqrt_total_counts_123 = sqrt(total_counts[0]) + sqrt(total_counts[1]) + sqrt(total_counts[2])
        sum_sqrt_total_counts_567 = sqrt(total_counts[4]) + sqrt(total_counts[5]) + sqrt(total_counts[6])

        sum_term_counts_123 = term_counts[0] + term_counts[1] + term_counts[2]
        sum_term_counts_567 = term_counts[4] + term_counts[5] + term_counts[6]

        active_period_trend = (sum_term_counts_567 / sum_sqrt_total_counts_567
                               ) - (sum_term_counts_123 / sum_sqrt_total_counts_123)

        recent_trend = 10 * (
                (term_counts[5] + term_counts[6]) / (sqrt(total_counts[5]) + sqrt(total_counts[6]))
                - (term_counts[3] + term_counts[4]) / (sqrt(total_counts[3]) + sqrt(total_counts[4])))

        mid_year_to_last_year_slope = 10 * (
                (term_counts[6] / sqrt(total_counts[6])) - (term_counts[3] / sqrt(total_counts[3]))) / 3

        return 2 * active_period_trend + mid_year_to_last_year_slope + recent_trend

    def escore2(self, show=False, term = None):

        xdata = np.linspace(0, self.NUM_PERIODS - 1, self.NUM_PERIODS)

        normalized_all = normalize(self.__per_period_counts_all)
        normalized_term = self.__per_period_counts_term

        trend = np.polyfit(xdata, normalized_term, 2)
        y_fit = trend[2] + (trend[1] * xdata) + (trend[0] * xdata * xdata)
        y_der = (trend[0] * xdata * 2) + trend[1]

        if show:
            plt.plot(xdata, normalized_term, 'o')
            plt.plot(xdata, y_fit)
            plt.plot(xdata, y_der)
            plt.legend(('term trend', 'fit curve', 'fit curve gradient'),
                       loc='upper left')
            if term is not None:
                plt.title('Term ' + term + " trend")

            plt.xlabel('quarter number')
            plt.ylabel('normalized frequency')


            plt.show()

            print("quadratic: " + str(fit_score(normalized_term, y_fit)))
            score = fit_score(normalized_term, y_fit)
        return  trend[0] if abs(trend[0]) >= 0.001 else trend[1]

    @staticmethod
    def escore_exponential(weekly_values, power=1):
        '''exponential like emergence score
        Description
            An emergence score designed to favour exponential like emergence,
            based on a yearly weighting function that linearly (power=1) increases from zero
        Arguments:
            weekly_values = list containing counts of patents occurring in each weekly period
            power = power of yearly weighting function (linear = 1)
        Returns:
            escore = emergence score
        Examples:
            escore = 1 all yearly_values in the last year
            escore = 2/3 yearly_values linearly increase from zero over 3 years (7/15 over 6 years, 0.5 infinite years)
            escore = 0 yearly_values equally spread over all years (horizontal line)
            escore = -2/3 yearly_values linearly decrease to zero over 3 years (-7/15 over 6 years, -0.5 infinite years)
            escore = -1 all yearly_values in the first year
        '''
        # todo: Modify not to use weekly values from self?
        # todo: Create -exp parameter, e.g. power of weight function
        # todo: Consider fractions or multiples of yearly values (effectively weeks per year different to 52)

        # convert into whole years, ending with last weekly value
        my_weekly_values = weekly_values.copy()
        weeks_in_year = 52  # use 52.1775 for mean weeks per calendar year
        num_whole_years = int(len(my_weekly_values) // weeks_in_year)
        my_weekly_values = my_weekly_values[-int(num_whole_years * weeks_in_year):]

        # calculate yearly values from weekly values
        yearly_values = []
        first_week_idx = 0
        for year in range(num_whole_years):
            # last_week_idx more complex if weeks_in_year is a float not integer
            last_week_idx = first_week_idx \
                            + int((num_whole_years - year) * weeks_in_year) \
                            - int((num_whole_years - year -1) * weeks_in_year)
            weekly_values_in_this_year = my_weekly_values[first_week_idx:last_week_idx]
            yearly_values.append(sum(weekly_values_in_this_year))
            first_week_idx = last_week_idx

        # escore = weighted yearly values / mean weighted yearly values
        yearly_weights = [x ** power for x in range(0, num_whole_years)]
        sum_weighted_yearly_values = sum(np.multiply(yearly_values, yearly_weights))
        sum_mean_weighted_yearly_values = sum(yearly_values) * np.mean(yearly_weights)
        try:
            # adjust score so that 0 instead of 1 gives a horizontal line (stationary)
            escore = sum_weighted_yearly_values / sum_mean_weighted_yearly_values - 1
        except:
            escore = 0
        return escore

    def escore_sigm(self, show=False, term=None):
        xdata = np.linspace(1, self.NUM_PERIODS_ACTIVE + self.NUM_PERIODS_BASE,
                            self.NUM_PERIODS_ACTIVE + self.NUM_PERIODS_BASE)
        ydata = self.__per_period_counts_term

        miny = min(ydata)
        maxy = max(ydata)
        diff = (maxy - miny)

        normalized_y = [(_y - miny) / diff for _y in ydata]

        popt, pcov = curve_fit(fsigmoid, xdata, normalized_y, maxfev=5000)
        print(popt)

        x = xdata
        y = [fsigmoid(x_, popt[0], popt[1]) for x_ in xdata]
        y_dev = [fsigmoid_derivative(x_, popt[0], popt[1]) for x_ in xdata]

        if show:
            plt.plot(xdata, normalized_y, 'o', label='data')
            plt.plot(x, y, label='fit')
            plt.plot(x, y_dev, label='deriv')
            plt.ylim(-0.2, 1)
            plt.legend(('term trend', 'fit curve', 'fit curve gradient'),
                       loc='upper left')
            if term is not None:
                plt.title('Term ' + term + " trend")

            plt.xlabel('quarter number')
            plt.ylabel('normalized frequency')
            plt.show()

            print("sigmoid: " + str(fit_score(normalized_y, y)))
        return fit_score(normalized_y, y), y_dev, y[len(y) - 1], y[0]
