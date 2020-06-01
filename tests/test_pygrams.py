import os
import unittest
from unittest import mock
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd
import pygrams
from scripts import FilePaths
from scripts.text_processing import WordAnalyzer
from scripts.utils.pygrams_exception import PygramsException


def bz2file_fake(file_name, state):
    assert state == 'wb', "Only supports file.open in write mode"
    m = MagicMock()
    m.__enter__.return_value = Mock()
    m.__exit__.return_value = Mock()
    m.__enter__.return_value = file_name
    return m


class TestPyGrams(unittest.TestCase):
    data_source_name = 'dummy.pkl.bz2'
    out_name = 'out'

    def setUp(self):
        self.global_stopwords = '''the
'''
        self.ngram_stopwords = '''patent with extra'''
        self.unigram_stopwords = '''of
'''

    def assertListAlmostEqual(self, list_a, list_b, places=7):
        self.assertEqual(len(list_a), len(list_b), 'Lists must be same length')
        for a, b in zip(list_a, list_b):
            self.assertAlmostEqual(a, b, places=places)

    def preparePyGrams(self, fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile):

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
            fake_df_data['publication_date'] = [
                f"{pd.Timestamp('2000-12-28 00:00:00') - pd.DateOffset(weeks=row):%Y-%m-%d}" for row
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
                m.__enter__.return_value.readlines.return_value = self.ngram_stopwords.split('\n')
                return m

            elif file_name == FilePaths.unigram_stopwords_filename:
                m.__enter__.return_value.read.return_value = self.unigram_stopwords
                return m

            else:
                return None

        mock_open.side_effect = open_fake_file

        mock_bz2file.side_effect = bz2file_fake

        def isfile_fake(file_name):
            if file_name == os.path.join('data', self.data_source_name):
                return True
            else:
                return False

        mock_path_isfile.side_effect = isfile_fake

    def assertTfidfOutputs(self, assert_func, mock_pickle_dump, mock_makedirs, max_df, min_date=200052,
                           max_date=200052):
        self.assertTrue(self.publication_date_auto_tested)
        self.assertTrue(self.patent_id_auto_tested)

        mock_makedirs.assert_called_with(self.tfidfOutputFolder(self.out_name, max_df, min_date, max_date),
                                         exist_ok=True)

        results_checked = False
        expected_tfidf_file_name = self.tfidfFileName(self.out_name, max_df, min_date, max_date)
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == expected_tfidf_file_name:
                tfidf_obj = dump_args[0][0]
                assert_func(tfidf_matrix=tfidf_obj.tfidf_matrix, feature_names=tfidf_obj.feature_names)

                results_checked = True
                break

        if not results_checked:
            self.fail('TFIDF results were not matched - were filenames correct?')

    def assertTimeSeriesOutputs(self, assert_func, mock_pickle_dump, mock_makedirs):
        self.assertTrue(self.publication_date_auto_tested)
        self.assertTrue(self.patent_id_auto_tested)

        output_folder_name = self.out_name + '-mdf-1.0-200052-200052/'
        expected_term_counts_filename = self.termCountsFileName(output_folder_name, self.out_name)

        results_checked = False
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == expected_term_counts_filename:
                [term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates] = dump_args[0][0]

                assert_func(term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates)

                results_checked = True
                break

        if not results_checked:
            self.fail('Term counts results were not matched - were filenames correct?')

    @staticmethod
    def tfidfOutputFolder(data_source_name, max_df, min_date, max_date):
        return os.path.join('cached', data_source_name + f'-mdf-{max_df}-{min_date}-{max_date}')

    @staticmethod
    def tfidfFileName(data_source_name, max_df, min_date, max_date):
        return os.path.join(TestPyGrams.tfidfOutputFolder(data_source_name, max_df, min_date, max_date),
                            'tfidf.pkl.bz2')

    @staticmethod
    def termCountsOutputFolder(dir_name):
        return os.path.join('outputs',dir_name, 'termcounts')

    @staticmethod
    def termCountsFileName(dir_name, name):
        return os.path.join(TestPyGrams.termCountsOutputFolder(dir_name), name + '-term_counts.pkl.bz2')

    @staticmethod
    def find_matching_pickle(mock_pickle_dump, pickle_file_name):
        for args in mock_pickle_dump.call_args_list:
            if args[0][1] == pickle_file_name:
                return args[0][0]
        return None

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("scripts.utils.utils.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("scripts.utils.utils.BZ2File", create=True)
    @mock.patch("scripts.utils.utils.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_output_tfidf(self, mock_path_isfile, mock_makedirs, mock_bz2file, mock_open, mock_pickle_dump,
                                 mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract'
            ]
        }
        max_df = 1.0
        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header', 'publication_date', '--max_document_frequency',
                str(max_df)]

        pygrams.main(args)

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            self.assertEqual(tfidf_matrix.todense(), np.ones(shape=(1, 1)), 'TFIDF should be 1x1 matrix of 1')
            self.assertListEqual(feature_names, ['abstract'])

        self.assertTfidfOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs, max_df)

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("scripts.utils.utils.read_pickle", create=True)
    @mock.patch("scripts.utils.utils.dump", create=True)
    @mock.patch("scripts.utils.utils.BZ2File", create=True)
    @mock.patch("scripts.utils.utils.makedirs", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    @mock.patch("scripts.output_factory.dump", create=True)
    @mock.patch("scripts.output_factory.BZ2File", create=True)
    @mock.patch("scripts.output_factory.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_output_to_cache_then_use_cache(self, mock_path_isfile,
                                                                             mock_output_makedirs,
                                                                             mock_output_bz2file,
                                                                             mock_output_pickle_dump,
                                                                             mock_output_open,
                                                                             mock_utils_makedirs,
                                                                             mock_utils_bz2file,
                                                                             mock_utils_pickle_dump,
                                                                             mock_utils_read_pickle,
                                                                             mock_open,
                                                                             mock_factory_read_pickle
                                                                             ):
        fake_df_data = {
            'abstract': [
                'abstract'
            ]
        }

        fake_output_file_content = {}

        def open_fake_output_file(file_name, state):
            self.assertEqual(state, 'w', "Only supports file.open in write mode")

            def snag_results(text):
                fake_output_file_content[file_name] = fake_output_file_content.get(file_name, '') + text

            m = MagicMock()
            m.__enter__.return_value = Mock()
            m.__enter__.return_value.write.side_effect = snag_results
            m.__exit__.return_value = Mock()
            return m

        mock_output_open.side_effect = open_fake_output_file

        # Make a note of the dumped TFIDF object for later
        self.preparePyGrams(fake_df_data, mock_factory_read_pickle, mock_open, mock_utils_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header', 'publication_date', '--max_document_frequency', '1.0']
        pygrams.main(args)

        # reset static object
        WordAnalyzer.tokenizer = None
        WordAnalyzer.preprocess = None
        WordAnalyzer.ngram_range = None
        WordAnalyzer.stemmed_stop_word_set_n = None
        WordAnalyzer.stemmed_stop_word_set_uni = None

        fake_output_file_content = {}

        # Fail if original data frame is requested from disc
        def factory_read_pickle_fake(pickle_file_name):
            self.fail(f'Should not be reading {pickle_file_name} via a factory if TFIDF was requested from pickle')

        dumped_tfidf_file_name = os.path.join('cached', self.out_name + '-mdf-1.0-200052-200052', 'tfidf.pkl.bz2')
        self.dumped_tfidf = self.find_matching_pickle(mock_utils_pickle_dump, dumped_tfidf_file_name)

        dumped_dates_file_name = os.path.join('cached', self.out_name + '-mdf-1.0-200052-200052', 'dates.pkl.bz2')
        self.dumped_dates = self.find_matching_pickle(mock_utils_pickle_dump, dumped_dates_file_name)

        dumped_cpc_dict_file_name = os.path.join('cached', self.out_name + '-mdf-1.0-200052-200052', 'cpc_dict.pkl.bz2')
        self.dumped_cpc_dict = self.find_matching_pickle(mock_utils_pickle_dump, dumped_cpc_dict_file_name)

        mock_factory_read_pickle.side_effect = factory_read_pickle_fake
        mock_utils_pickle_dump.reset_mock(return_value=True, side_effect=True)

        # Instead support TFIDF pickle read - and return the TFIDF object previously saved to disc
        def pipeline_read_pickle_fake(pickle_file_name):
            if pickle_file_name == dumped_tfidf_file_name:
                return self.dumped_tfidf
            elif pickle_file_name == dumped_dates_file_name:
                return self.dumped_dates
            elif pickle_file_name == dumped_cpc_dict_file_name:
                return self.dumped_cpc_dict
            else:
                self.fail(f'Should not be reading {pickle_file_name} via a factory if TFIDF was requested from pickle')

        mock_output_bz2file.side_effect = bz2file_fake
        mock_utils_read_pickle.side_effect = pipeline_read_pickle_fake
        mock_utils_read_pickle.return_value = self.dumped_tfidf
        args = ['-ds', self.data_source_name, '-ts',
                '--date_header',
                'publication_date', '--max_document_frequency', '1.0',
                '--use_cache', self.out_name + '-mdf-1.0-200052-200052']
        pygrams.main(args)

        self.assertEqual(' abstract                       1.000000\n',
                         fake_output_file_content[
                             os.path.join('outputs', self.out_name+'-mdf-1.0-200052-200052', 'reports',
                                          self.out_name+'_keywords.txt')])

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("scripts.utils.utils.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("scripts.utils.utils.BZ2File", create=True)
    @mock.patch("scripts.utils.utils.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_two_patents_unigrams_only_output_tfidf(self, mock_path_isfile, mock_makedirs, mock_bz2file,
                                                           mock_open, mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract one',
                'abstract two'
            ]
        }
        max_df = 1.0

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header',
                'publication_date', '--max_document_frequency', str(max_df), '--max_ngrams', '1']

        pygrams.main(args)

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

        self.assertTfidfOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs, max_df, 200051, 200052)

    """
    Extended from test_simple_two_patents_unigrams_only_output_tfidf - sets prefilter-terms to remove 'noise' terms
    """
    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("scripts.utils.utils.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("scripts.utils.utils.BZ2File", create=True)
    @mock.patch("scripts.utils.utils.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_two_patents_unigrams_and_prefilter_only_output_tfidf(self, mock_path_isfile, mock_makedirs,
                                                                         mock_bz2file, mock_open, mock_pickle_dump,
                                                                         mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract one',
                'abstract two'
            ]
        }
        max_df = 1.0
        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header', 'publication_date',
                '--max_document_frequency', str(max_df), '--max_ngrams', '1',
                '--prefilter_terms', '1']

        pygrams.main(args)

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

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            self.assertListEqual(feature_names, ['abstract', 'one'])
            tfidf_as_lists = tfidf_matrix.todense().tolist()
            self.assertListAlmostEqual([tfidf_as_lists[0][0]], [l2norm_tfidf_abstract], places=4)
            self.assertListAlmostEqual([tfidf_as_lists[1][0]], [l2norm_tfidf_abstract], places=4)

        self.assertTfidfOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs, max_df, 200051, 200052)

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("scripts.utils.utils.dump", create=True)
    @mock.patch("scripts.utils.utils.BZ2File", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("scripts.output_factory.dump", create=True)
    @mock.patch("scripts.output_factory.BZ2File", create=True)
    @mock.patch("scripts.output_factory.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_unibitri_reduction_output_termcounts(self, mock_path_isfile, mock_of_makedirs,
                                                  mock_of_bz2file, mock_of_dump, mock_open,
                                                  mock_utils_bz2file, mock_utils_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract 1, of the patent with extra stuff'
            ]
        }

        mock_of_bz2file.side_effect = bz2file_fake

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_utils_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--id_header', 'patent_id', '--date_header',
                'publication_date', '--max_document_frequency', '1.0']  # '-ts', '-tc'

        pygrams.main(args)

        dumped_tfidf_file_name = os.path.join('cached', self.out_name + '-mdf-1.0-200052-200052', 'tfidf.pkl.bz2')
        self.dumped_tfidf = self.find_matching_pickle(mock_utils_dump, dumped_tfidf_file_name)

        dumped_dates_file_name = os.path.join('cached', self.out_name + '-mdf-1.0-200052-200052', 'dates.pkl.bz2')
        self.dumped_dates = self.find_matching_pickle(mock_utils_dump, dumped_dates_file_name)

        self.assertListEqual(self.dumped_tfidf.feature_names, ['abstract', 'of patent with', 'with extra stuff'])
        term_counts_as_lists = self.dumped_tfidf.count_matrix.todense().tolist()
        self.assertListEqual(term_counts_as_lists[0], [1, 1, 1])
        self.assertListEqual(self.dumped_dates.tolist(), [200052])

    @unittest.skip("json compulsory now, so not an option")
    def test_args_json_not_requested(self):
        args = pygrams.get_args([])
        self.assertFalse(args.json)

    @unittest.skip("json compulsory now, so not an option")
    def test_args_json_requested_short(self):
        args = pygrams.get_args(['-j'])
        self.assertTrue(args.json)

    @unittest.skip("json compulsory now, so not an option")
    def test_args_json_requested_long(self):
        args = pygrams.get_args(['--json'])
        self.assertTrue(args.json)

    def test_args_output_name_requested_long(self):
        args = pygrams.get_args(['--outputs_name=my/test/name.txt'])
        self.assertEqual('my/test/name.txt', args.outputs_name)

    def test_args_document_source_requested_long(self):
        args = pygrams.get_args(['--doc_source=my-test'])
        self.assertEqual('my-test', args.doc_source)

    @mock.patch("scripts.output_factory.json.dump", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    def test_json_configuration_encoding_sum_no_time_weighting(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'USPTO-random-100.pkl.bz2'
        patent_pickle_absolute_file_name = os.path.abspath(os.path.join('data', patent_pickle_file_name))
        output_file_name = 'test'
        suffix = '-mdf-0.05-200502-201808'
        report_file_name = os.path.join('outputs',output_file_name+suffix, 'json_config', output_file_name + '_keywords.txt')
        json_file_name = os.path.join('outputs',output_file_name+suffix, 'json_config', output_file_name + '_keywords_config.json')
        pygrams.main([f'--outputs_name={output_file_name}', '-f=set', '-p=sum', '-cpc=Y12',
                      '--date_from=1999/03/12', '--date_to=2000/11/30', '-dh', 'publication_date', '-ds',
                      patent_pickle_file_name])

        mock_open.assert_any_call(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': 199910,
                'to': 200048
            },
            'parameters': {
                'pick': 'sum'
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.output_factory.json.dump", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    def test_json_configuration_encoding_maximal(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'USPTO-random-100.pkl.bz2'
        patent_pickle_absolute_file_name = os.path.abspath(os.path.join('data', patent_pickle_file_name))
        output_file_name = 'test'
        report_file_name = os.path.join('outputs', 'test-mdf-0.05-200502-201808', 'json_config', output_file_name + '_keywords.txt')
        json_file_name = os.path.join('outputs', 'test-mdf-0.05-200502-201808','json_config', output_file_name + '_keywords_config.json')
        pygrams.main([f'--outputs_name={output_file_name}', '-p=max', '-cpc=Y12',
                      '--date_from=1998/01/01', '--date_to=2001/12/31', '-dh', 'publication_date', '-ds',
                      patent_pickle_file_name])

        mock_open.assert_any_call(json_file_name, 'w')
        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': 199801,
                'to': 200201
            },
            'parameters': {
                'pick': 'max'
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("os.path.isfile", create=True)
    def test_reports_unsupported_df_format(self, mock_path_isfile):

        unknown_filename = 'unknown.format'

        def isfile_fake(file_name):
            if file_name == os.path.join('data', unknown_filename):
                return True
            else:
                return False

        mock_path_isfile.side_effect = isfile_fake
        test_args = ['--doc_source', unknown_filename]
        try:
            pygrams.main(test_args)
            self.fail("should raise exception")
        except PygramsException as err:
            self.assertEqual('Unsupported file: ' + os.path.join('data', unknown_filename), err.message)


if __name__ == '__main__':
    unittest.main()
