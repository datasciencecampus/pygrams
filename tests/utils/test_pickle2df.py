import unittest

import pandas as pd

from scripts.utils.pickle2df import PatentsPickle2DataFrame


class TestPatentsPickle2DataFrame(unittest.TestCase):
    test_pickle_name = 'test pickle'

    @staticmethod
    def pickle_reader(pickle_name):
        if pickle_name == TestPatentsPickle2DataFrame.test_pickle_name:
            return pd.DataFrame({
                'publication_date': [pd.Timestamp('2007-07-08 00:00:00'), pd.Timestamp('2003-01-21 00:00:00'),
                                     pd.Timestamp('2016-12-31 00:00:00')],
                'classifications_cpc': [['Q99Q 123/456', 'Y02L 238/7209'], ['H01L  24/03'], ['Y02L2224/85203']],
                'invention_title': ['patent 1', 'patent 2', 'patent 3']
            })
        else:
            return None

    def test_filters_by_date_no_effect_if_None(self):
        date_from = None
        date_to = None
        self.print_output = ''

        def print_func(t):
            self.print_output += t

        data_frame = PatentsPickle2DataFrame(self.test_pickle_name, date_from, date_to,
                                             pickle_reader=self.pickle_reader,
                                             print_func=print_func).data_frame

        self.assertEqual(3, data_frame.shape[0])
        self.assertEqual('patent 1', data_frame.loc[0].invention_title)
        self.assertEqual('patent 2', data_frame.loc[1].invention_title)
        self.assertEqual('patent 3', data_frame.loc[2].invention_title)
        self.assertEqual('', self.print_output)

    def test_filters_by_to_date_if_not_None(self):
        date_from = '1999-01-01'
        date_to = '2007-07-07'
        self.print_output = ''

        def print_func(t):
            self.print_output += t + '\n'

        data_frame = PatentsPickle2DataFrame(self.test_pickle_name, date_from, date_to,
                                             pickle_reader=self.pickle_reader,
                                             print_func=print_func).data_frame

        self.assertEqual(1, data_frame.shape[0])
        self.assertEqual('patent 2', data_frame.loc[0].invention_title)
        self.assertEqual('Sifting documents between 01-Jan-1999 and 07-Jul-2007\n'
                         '1 documents available after date sift\n', self.print_output)

    def test_filters_by_from_date_if_not_None(self):
        date_from = '2003-01-22'
        date_to = '2020-02-01'
        self.print_output = ''

        def print_func(t):
            self.print_output += t + '\n'

        data_frame = PatentsPickle2DataFrame(self.test_pickle_name, date_from, date_to,
                                             pickle_reader=self.pickle_reader,
                                             print_func=print_func).data_frame

        self.assertEqual(2, data_frame.shape[0])
        self.assertEqual('patent 1', data_frame.loc[0].invention_title)
        self.assertEqual('patent 3', data_frame.loc[1].invention_title)
        self.assertEqual('Sifting documents between 22-Jan-2003 and 01-Feb-2020\n'
                         '2 documents available after date sift\n', self.print_output)
