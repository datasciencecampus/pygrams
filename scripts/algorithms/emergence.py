from math import sqrt

import numpy as np
import pylab as plt
from scipy.optimize import curve_fit

from scripts.utils.utils import fsigmoid, fsigmoid_derivative, fit_score


class Emergence(object):

    def __init__(self, timeseries_global):
        self.BASE_TERM2ALL_RATIO_THRESHOLD = 0.15
        self.ACTIVE2BASE_RATIO_THRESHOLD = 2
        self.MIN_DOCS_FOR_EMERGENCE = 7

        self.NUM_PERIODS_BASE = 3
        self.NUM_PERIODS_ACTIVE = 7
        self.NUM_PERIODS = self.NUM_PERIODS_BASE + self.NUM_PERIODS_ACTIVE

        self.__timeseries_global = timeseries_global

        total_counts = self.__timeseries_global[-self.NUM_PERIODS_ACTIVE:]

        self.__sum_sqrt_total_counts_123 = sqrt(total_counts[0]) + sqrt(total_counts[1]) + sqrt(total_counts[2])
        self.__sum_sqrt_total_counts_567 = sqrt(total_counts[4]) + sqrt(total_counts[5]) + sqrt(total_counts[6])

    def is_emergence_candidate(self, timeseries_term):
        num_term_records = len(timeseries_term)

        num_records_base_all = sum(self.__timeseries_global[-self.NUM_PERIODS:-self.NUM_PERIODS_ACTIVE])
        num_records_base_term = sum(timeseries_term[-self.NUM_PERIODS:-self.NUM_PERIODS_ACTIVE])
        num_records_active_term = sum(timeseries_term[-self.NUM_PERIODS_ACTIVE:])
        if num_records_base_term == 0:
            return False

        base2all_below_threshold = num_records_base_term / num_records_base_all < self.BASE_TERM2ALL_RATIO_THRESHOLD
        at_least_n_recs = num_term_records >= self.MIN_DOCS_FOR_EMERGENCE
        active2base_ratio = num_records_active_term / num_records_base_term

        return at_least_n_recs and active2base_ratio > self.ACTIVE2BASE_RATIO_THRESHOLD and base2all_below_threshold \
               and self.has_multiple_author_sets()

    @staticmethod
    def has_multiple_author_sets():
        return True

    def calculate_escore(self, timeseries_term):
        timeseries_term_active = timeseries_term[-self.NUM_PERIODS_ACTIVE:]
        timeseries_global_active = self.__timeseries_global[-self.NUM_PERIODS_ACTIVE:]

        sum_term_counts_123 = timeseries_term_active[0] + timeseries_term_active[1] + timeseries_term_active[2]
        sum_term_counts_567 = timeseries_term_active[4] + timeseries_term_active[5] + timeseries_term_active[6]

        active_period_trend = (sum_term_counts_567 / self.__sum_sqrt_total_counts_567) - (
                sum_term_counts_123 / self.__sum_sqrt_total_counts_123)

        recent_trend = 10 * (
                (timeseries_term_active[5] + timeseries_term_active[6]) / (sqrt(timeseries_global_active[5]) + sqrt(timeseries_global_active[6]))
                - (timeseries_term_active[3] + timeseries_term_active[4]) / (sqrt(timeseries_global_active[3]) + sqrt(timeseries_global_active[4])))

        mid_year_to_last_year_slope = 10 * (
                (timeseries_term_active[6] / sqrt(timeseries_global_active[6])) - (timeseries_term_active[3] / sqrt(timeseries_global_active[3]))) / 3

        return 2 * active_period_trend + mid_year_to_last_year_slope + recent_trend

    @staticmethod
    def escore2(timeseries_term, show=False, term=None):

        xdata = np.linspace(0, len(timeseries_term) - 1, len(timeseries_term))
        normalized_term = timeseries_term

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
            score = fit_score(normalized_term, y_fit)
            print("quadratic: " + str(score))

        return trend[0]  # if abs(trend[0]) >= 0.001 else trend[1]

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

    @staticmethod
    def escore_exponential(weekly_values, power=1):
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
                            - int((num_whole_years - year - 1) * weeks_in_year)
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
        except ValueError:
            escore = 0
        return escore

    def escore_sigm(self, term_period_counts, show=False, term=None):
        xdata = np.linspace(1, self.NUM_PERIODS_ACTIVE + self.NUM_PERIODS_BASE,
                            self.NUM_PERIODS_ACTIVE + self.NUM_PERIODS_BASE)
        ydata = term_period_counts

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
