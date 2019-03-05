import os
import unittest
from unittest import mock
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd

import pygrams2
from scripts import FilePaths


class TestPipeline(unittest.TestCase):
    data_source_name = 'dummy.pkl.bz2'

    def setUp(self):
        self.global_stopwords = '''the
'''
        self.ngram_stopwords = '''with
'''
        self.unigram_stopwords = '''of
'''

    def assertListAlmostEqual(self, list_a, list_b, places=7):
        self.assertEqual(len(list_a), len(list_b), 'Lists must be same length')
        for a, b in zip(list_a, list_b):
            self.assertAlmostEqual(a, b, places=places)

    def preparePyGrams(self, fake_df_data, mock_read_pickle, mock_open, mock_bz2file):

        self.number_of_rows = len(fake_df_data['abstract'])
        self.patent_id_auto_tested = 'patent_id' not in fake_df_data
        self.application_id_auto_tested = 'application_id' not in fake_df_data
        self.application_date_auto_tested = 'application_date' not in fake_df_data
        self.publication_date_auto_tested = 'publication_date' not in fake_df_data
        self.invention_title_auto_tested = 'invention_title' not in fake_df_data
        self.classifications_cpc_auto_tested = 'classifications_cpc' not in fake_df_data
        self.inventor_names_auto_tested = 'inventor_names' not in fake_df_data
        self.inventor_countries_auto_tested = 'inventor_countries' not in fake_df_data
        self.inventor_cities_auto_tested = 'inventor_cities' not in fake_df_data
        self.applicant_organisation_auto_tested = 'applicant_organisation' not in fake_df_data
        self.applicant_countries_auto_tested = 'applicant_countries' not in fake_df_data
        self.applicant_cities_auto_tested = 'applicant_cities' not in fake_df_data

        if self.patent_id_auto_tested:
            fake_df_data['patent_id'] = [f'patent_id-{pid}' for pid in range(self.number_of_rows)]

        if self.application_id_auto_tested:
            fake_df_data['application_id'] = [f'application_id-{pid}' for pid in range(self.number_of_rows)]

        if self.application_date_auto_tested:
            fake_df_data['application_date'] = [pd.Timestamp('1998-01-01 00:00:00') + pd.DateOffset(weeks=row) for row
                                                in range(self.number_of_rows)]

        if self.publication_date_auto_tested:
            fake_df_data['publication_date'] = [pd.Timestamp('2000-12-28 00:00:00') - pd.DateOffset(weeks=row) for row
                                                in range(self.number_of_rows)]

        if self.invention_title_auto_tested:
            fake_df_data['invention_title'] = [f'invention_title-{pid}' for pid in range(self.number_of_rows)]

        if self.classifications_cpc_auto_tested:
            fake_df_data['classifications_cpc'] = [[f'Y{row:02}'] for row in range(self.number_of_rows)]

        if self.inventor_names_auto_tested:
            fake_df_data['inventor_names'] = [[f'Fred {row:02}'] for row in range(self.number_of_rows)]

        if self.inventor_countries_auto_tested:
            fake_df_data['inventor_countries'] = [['GB']] * self.number_of_rows

        if self.inventor_cities_auto_tested:
            fake_df_data['inventor_cities'] = [['Newport']] * self.number_of_rows

        if self.applicant_organisation_auto_tested:
            fake_df_data['applicant_organisation'] = [['Neat and tidy']] * self.number_of_rows

        if self.applicant_countries_auto_tested:
            fake_df_data['applicant_countries'] = [['GB']] * self.number_of_rows

        if self.applicant_cities_auto_tested:
            fake_df_data['applicant_cities'] = [['Newport']] * self.number_of_rows

        df = pd.DataFrame(data=fake_df_data)
        mock_read_pickle.return_value = df

        def open_fake_file(file_name, state):
            self.assertEqual(state, 'r', "Only supports file.open in read mode")
            m = MagicMock()
            m.__enter__.return_value = Mock()
            m.__exit__.return_value = Mock()

            if file_name == FilePaths.global_stopwords_filename:
                m.__enter__.return_value.read.return_value = self.global_stopwords
                return m

            elif file_name == FilePaths.ngram_stopwords_filename:
                m.__enter__.return_value.read.return_value = self.ngram_stopwords
                return m

            elif file_name == FilePaths.unigram_stopwords_filename:
                m.__enter__.return_value.read.return_value = self.unigram_stopwords
                return m

            else:
                return None

        mock_open.side_effect = open_fake_file

        def bz2file_fake(file_name, state):
            self.assertEqual(state, 'wb', "Only supports file.open in write mode")
            m = MagicMock()
            m.__enter__.return_value = Mock()
            m.__exit__.return_value = Mock()
            m.__enter__.return_value = file_name
            return m

        mock_bz2file.side_effect = bz2file_fake

    def assertOutputsExceptTfidfFeatureNames(self, assert_func, mock_pickle_dump, mock_makedirs):
        self.assertTrue(self.publication_date_auto_tested)
        self.assertTrue(self.patent_id_auto_tested)

        mock_makedirs.assert_called_with(self.tfidfOutputFolder(), exist_ok=True)
        results_checked = False
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == self.tfidfFileName(self.data_source_name):
                [tfidf_matrix, feature_names, document_week_dates, doc_ids] = dump_args[0][0]

                assert_func(tfidf_matrix=tfidf_matrix, feature_names=feature_names)
                self.assertListEqual(doc_ids, [f'patent_id-{pid}' for pid in range(self.number_of_rows)])
                self.assertListEqual(document_week_dates, [200052 - row for row in range(self.number_of_rows)])

                results_checked = True
                break

        if not results_checked:
            self.fail('Results were not matched - were filenames correct?')

    @staticmethod
    def tfidfFileName(data_source_name):
        return os.path.join('outputs', 'tfidf', data_source_name + '-tfidf.pkl.bz2')

    @staticmethod
    def tfidfOutputFolder():
        return os.path.join('outputs', 'tfidf')

    @mock.patch("pandas.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.tfidf_wrapper.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("os.makedirs", create=True)
    def test_simple_export_tfidf(self, mock_makedirs, mock_bz2file, mock_open, mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file)
        args = ['-o', 'tfidf', '-ds', self.data_source_name, '--id_header', 'patent_id', '--date_header',
                'publication_date', '--max_document_frequency', '1.0']

        return_value = pygrams2.main(args)

        self.assertEqual(0, return_value, 'Return value indicates failure')

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            self.assertEqual(tfidf_matrix.todense(), np.ones(shape=(1, 1)), 'TFIDF should be 1x1 matrix of 1')
            self.assertListEqual(feature_names, ['abstract'])

        self.assertOutputsExceptTfidfFeatureNames(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs)

    @mock.patch("pandas.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.tfidf_wrapper.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("os.makedirs", create=True)
    def test_simple_two_patents_unigrams_only_export_tfidf(self, mock_makedirs, mock_bz2file, mock_open,
                                                           mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract one',
                'abstract two'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file)
        args = ['-o', 'tfidf', '-ds', self.data_source_name, '--id_header', 'patent_id', '--date_header',
                'publication_date', '--max_document_frequency', '1.0', '--max_n', '1']

        return_value = pygrams2.main(args)

        self.assertEqual(0, return_value, 'Return value indicates failure')

        # tf(t) = num of occurrences / number of words in doc
        #
        # smoothing is false, so no modification to log numerator or denominator:
        # idf(d, t) = log [ n / df(d, t) ] + 1
        #
        # n = total number of docs
        #
        # norm='l2' by default

        tfidf_abstract = (1 / 2) * (np.log(2 / 2) + 1)
        tfidf_one = (1 / 2) * (np.log(2 / 1) + 1)
        l2norm = np.sqrt(tfidf_abstract * tfidf_abstract + tfidf_one * tfidf_one)
        l2norm_tfidf_abstract = tfidf_abstract / l2norm
        l2norm_tfidf_one = tfidf_one / l2norm

        # Note that 'one' will have same weight as 'two' given where it appears

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            self.assertListEqual(feature_names, ['abstract', 'one', 'two'])
            tfidf_as_lists = tfidf_matrix.todense().tolist()
            self.assertListAlmostEqual(tfidf_as_lists[0], [l2norm_tfidf_abstract, l2norm_tfidf_one, 0], places=4)
            self.assertListAlmostEqual(tfidf_as_lists[1], [l2norm_tfidf_abstract, 0, l2norm_tfidf_one], places=4)

        self.assertOutputsExceptTfidfFeatureNames(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs)

    @mock.patch("pandas.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("sscripts.tfidf_wrapper.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("os.makedirs", create=True)
    def test_stopwords_export_tfidf(self, mock_makedirs, mock_bz2file, mock_open, mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract 1, of the patent with extra stuff'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file)
        args = ['-o', 'tfidf', '-ds', self.data_source_name, '--id_header', 'patent_id', '--date_header',
                'publication_date', '--max_document_frequency', '1.0']

        return_value = pygrams2.main(args)

        self.assertEqual(0, return_value, 'Return value indicates failure')

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            tfidf_as_lists = tfidf_matrix.todense().tolist()
            self.assertListEqual(feature_names, ['abstract', 'extra', 'extra stuff', 'patent', 'stuff', 'with'])
            self.assertListAlmostEqual(tfidf_as_lists[0], [0.4082, 0, 0.4082, 0.4082, 0, 0.4082], places=4)

        self.assertOutputsExceptTfidfFeatureNames(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs)


if __name__ == '__main__':
    unittest.main()
