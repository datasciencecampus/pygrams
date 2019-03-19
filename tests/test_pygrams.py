import os
import unittest
from unittest import mock
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd

import pygrams
from scripts import FilePaths
from scripts.utils.pygrams_exception import PygramsException


class TestPyGrams(unittest.TestCase):
    data_source_name = 'dummy.pkl.bz2'
    out_name = 'out'

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

        def isfile_fake(file_name):
            if file_name == os.path.join('data', self.data_source_name):
                return True
            else:
                return False

        mock_path_isfile.side_effect = isfile_fake

    def assertTfidfOutputs(self, assert_func, mock_pickle_dump, mock_makedirs):
        self.assertTrue(self.publication_date_auto_tested)
        self.assertTrue(self.patent_id_auto_tested)

        mock_makedirs.assert_called_with(self.tfidfOutputFolder(), exist_ok=True)
        results_checked = False
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == self.tfidfFileName(self.out_name):
                tfidf_pickle = dump_args[0][0]
                tfidf_obj = tfidf_pickle[0]

                assert_func(tfidf_matrix=tfidf_obj.tfidf_matrix, feature_names=tfidf_obj.feature_names)

                results_checked = True
                break

        if not results_checked:
            self.fail('TFIDF results were not matched - were filenames correct?')

    def assertTermCountOutputs(self, assert_func, mock_pickle_dump, mock_makedirs):
        self.assertTrue(self.publication_date_auto_tested)
        self.assertTrue(self.patent_id_auto_tested)

        # mock_makedirs.assert_called_with(self.termCountsOutputFolder(), exist_ok=True)
        results_checked = False
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == self.termCountsFileName(self.out_name):
                [term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates] = dump_args[0][0]

                assert_func(term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates)

                results_checked = True
                break

        if not results_checked:
            self.fail('Term counts results were not matched - were filenames correct?')

    @staticmethod
    def tfidfOutputFolder():
        return os.path.join('outputs', 'tfidf')

    @staticmethod
    def tfidfFileName(data_source_name):
        return os.path.join(TestPyGrams.tfidfOutputFolder(), data_source_name + '-tfidf.pkl.bz2')

    @staticmethod
    def termCountsOutputFolder():
        return os.path.join('outputs', 'termcounts')

    @staticmethod
    def termCountsFileName(data_source_name):
        return os.path.join(TestPyGrams.termCountsOutputFolder(), data_source_name + '-term_counts.pkl.bz2')

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("scripts.pipeline.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_output_tfidf(self, mock_path_isfile, mock_makedirs, mock_bz2file, mock_open, mock_pickle_dump,
                                 mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header', 'publication_date', '--max_document_frequency', '1.0']

        pygrams.main(args)

        def assert_tfidf_outputs(tfidf_matrix, feature_names):
            self.assertEqual(tfidf_matrix.todense(), np.ones(shape=(1, 1)), 'TFIDF should be 1x1 matrix of 1')
            self.assertListEqual(feature_names, ['abstract'])

        self.assertTfidfOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs)

    @mock.patch("scripts.pipeline.read_pickle", create=True)
    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("scripts.pipeline.makedirs", create=True)
    @mock.patch("scripts.output_factory.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_output_tfidf_pickle_and_unpickle(self, mock_path_isfile, mock_output_makedirs,
                                                     mock_pipeline_makedirs, mock_bz2file, mock_open,
                                                     mock_pickle_dump,
                                                     mock_factory_read_pickle, mock_pipeline_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract'
            ]
        }

        # Make a note of the dumped TFIDF object for later
        self.preparePyGrams(fake_df_data, mock_factory_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header', 'publication_date', '--max_document_frequency', '1.0']
        pygrams.main(args)

        # Fail if original data frame is requested from disc
        def factory_read_pickle_fake(pickle_file_name):
            self.fail(f'Should not be reading {pickle_file_name} via a factory if TFIDF was requested from pickle')

        self.dumped_tfidf_obj = mock_pickle_dump.call_args_list[0][0][0]
        mock_factory_read_pickle.side_effect = factory_read_pickle_fake
        mock_pickle_dump.reset_mock(return_value=True, side_effect=True)

        # Instead support TFIDF pickle read - and return the TFIDF object previously saved to disc
        def pipeline_read_pickle_fake(pickle_file_name):
            if pickle_file_name == os.path.join('outputs','tfidf', self.out_name + '-tfidf.pkl.bz2'):
                return self.dumped_tfidf_obj
            self.fail(f'Should not be reading {pickle_file_name} via a factory if TFIDF was requested from pickle')

        mock_pipeline_read_pickle.side_effect = pipeline_read_pickle_fake
        mock_pipeline_read_pickle.return_value = self.dumped_tfidf_obj
        args = ['-tc', '-ds', self.data_source_name, '--date_header',
                'publication_date', '--max_document_frequency', '1.0',
                '--input_tfidf', self.out_name + '-tfidf.pkl.bz2']
        pygrams.main(args)

        def assert_tfidf_outputs(term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates):
            self.assertEqual(term_counts_per_week.todense(), np.ones(shape=(1, 1)),
                             'term counts should be 1x1 matrix of 1')
            self.assertListEqual(feature_names, ['abstract'])
            self.assertListEqual(number_of_documents_per_week, [1])
            self.assertListEqual(week_iso_dates, [200052])

        self.assertTermCountOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_output_makedirs)

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("scripts.pipeline.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_simple_two_patents_unigrams_only_output_tfidf(self, mock_path_isfile, mock_makedirs, mock_bz2file,
                                                           mock_open, mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract one',
                'abstract two'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-ds', self.data_source_name, '--date_header',
                'publication_date', '--max_document_frequency', '1.0', '--max_ngrams', '1']

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

        self.assertTfidfOutputs(assert_tfidf_outputs, mock_pickle_dump, mock_makedirs)

    @mock.patch("scripts.data_factory.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.text_processing.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("scripts.output_factory.makedirs", create=True)
    @mock.patch("os.path.isfile", create=True)
    def test_unibitri_reduction_output_termcounts(self, mock_path_isfile, mock_makedirs, mock_bz2file, mock_open,
                                                  mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract 1, of the patent with extra stuff'
            ]
        }

        self.preparePyGrams(fake_df_data, mock_read_pickle, mock_open, mock_bz2file, mock_path_isfile)
        args = ['-tc', '-ds', self.data_source_name, '--id_header', 'patent_id', '--date_header',
                'publication_date', '--max_document_frequency', '1.0']

        pygrams.main(args)

        def assert_outputs(term_counts_per_week, feature_names, number_of_documents_per_week, week_iso_dates):
            self.assertListEqual(feature_names, ['abstract', 'extra', 'extra stuff', 'patent', 'stuff', 'with'])
            term_counts_as_lists = term_counts_per_week.todense().tolist()
            self.assertListEqual(term_counts_as_lists[0], [1, 0, 1, 1, 0, 1])
            self.assertListEqual(number_of_documents_per_week, [1])
            self.assertListEqual(week_iso_dates, [200052])

        self.assertTermCountOutputs(assert_outputs, mock_pickle_dump, mock_makedirs)

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
        report_file_name = os.path.join('outputs', 'reports', output_file_name + '.txt')
        json_file_name = os.path.join('outputs', 'reports', output_file_name + '.json')
        pygrams.main([f'--outputs_name={output_file_name}', '-f=set', '-p=sum', '-cpc=Y12',
                      '--date_from=1999/03/12', '--date_to=2000/11/30', '-dh', 'publication_date', '-ds', patent_pickle_file_name])

        mock_open.assert_any_call(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': '1999-03-12',
                'to': '2000-11-30'
            },
            'parameters': {
                'pick': 'sum',
                'time': False
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.output_factory.json.dump", create=True)
    @mock.patch("scripts.output_factory.open", create=True)
    def test_json_configuration_encoding_maximal_and_time_weighting(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'USPTO-random-100.pkl.bz2'
        patent_pickle_absolute_file_name = os.path.abspath(os.path.join('data', patent_pickle_file_name))
        output_file_name = 'test'
        report_file_name = os.path.join('outputs', 'reports', output_file_name + '.txt')
        json_file_name = os.path.join('outputs', 'reports', output_file_name + '.json')
        pygrams.main([ f'--outputs_name={output_file_name}', '-t', '-p=max', '-cpc=Y12',
                      '--date_from=1998/01/01', '--date_to=2001/12/31', '-dh', 'publication_date', '-ds', patent_pickle_file_name])

        mock_open.assert_any_call(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_absolute_file_name,
                'tech_report': report_file_name
            },
            'month_year': {
                'from': '1998-01-01',
                'to': '2001-12-31'
            },
            'parameters': {
                'pick': 'max',
                'time': True
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("scripts.terms_graph.json.dump", create=True)
    @mock.patch("scripts.terms_graph.open", create=True)
    def test_graph_creation(self, mock_open, mock_json_dump):
        fname = 'other'
        js_file_name = os.path.join('outputs', 'visuals', 'key-terms.js')
        json_file_name = os.path.join('outputs', 'reports', 'key-terms.json')
        graph_report_name = os.path.join('outputs', 'reports', fname + '_graph.txt')

        test_args = ['--doc_source', 'USPTO-random-100.pkl.bz2', '-o', 'graph', '--outputs_name', fname]
        pygrams.main(test_args)

        mock_open.assert_any_call(json_file_name, 'w')
        mock_open.assert_any_call(js_file_name, 'w')
        mock_open.assert_any_call(graph_report_name, 'w')

        actual_json = mock_json_dump.call_args_list[0][0][0]
        self.assertIn('nodes', actual_json)
        self.assertIn('links', actual_json)

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
