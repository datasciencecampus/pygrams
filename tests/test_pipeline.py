import os
import unittest
from unittest import mock
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd

import pygrams
from scripts import FilePaths


class TestPipeline(unittest.TestCase):

    @mock.patch("pandas.read_pickle", create=True)
    @mock.patch("pickle.dump", create=True)
    @mock.patch("scripts.algorithms.tfidf.open", create=True)
    @mock.patch("bz2.BZ2File", create=True)
    @mock.patch("os.makedirs", create=True)
    def test_export_tfidf(self, mock_makedirs, mock_bz2file, mock_open, mock_pickle_dump, mock_read_pickle):
        fake_df_data = {
            'abstract': [
                'abstract 1'
            ],
            'patent_id': [
                'family0'
            ],
            'application_id': [
                'app_orig0'
            ],
            'application_date': [
                pd.Timestamp('1998-01-01 00:00:00')
            ],
            'publication_date': [
                pd.Timestamp('1999-01-08 00:00:00')
            ],
            'invention_title': [
                'title1'
            ],
            'classifications_cpc': [
                'Y02'
            ],
            'inventor_names': [
                ['A N OTHER', 'JONES FRED']
            ],
            'inventor_countries': [
                ['US', 'DE']
            ],
            'inventor_cities': [
                ['Ontario N2L 3W8', '73527 Schwäbisch Gmünd']
            ],
            'applicant_organisation': [
                ['FISH I AM']
            ],
            'applicant_countries': [
                ['GB']
            ],
            'applicant_cities': [
                ['Newport NP20 1XJ']
            ]
        }
        df = pd.DataFrame(data=fake_df_data)
        mock_read_pickle.return_value = df

        global_stopwords = '''the
other
'''
        ngram_stopwords = '''with
'''
        unigram_stopwords = '''unusually
because
'''

        def open_fake_file(file_name, state):
            m = MagicMock()
            m.__enter__.return_value = Mock()
            m.__exit__.return_value = Mock()

            if file_name == FilePaths.global_stopwords_filename:
                m.__enter__.return_value.read.return_value = global_stopwords
                return m

            elif file_name == FilePaths.ngram_stopwords_filename:
                m.__enter__.return_value.read.return_value = ngram_stopwords
                return m

            elif file_name == FilePaths.unigram_stopwords_filename:
                m.__enter__.return_value.read.return_value = unigram_stopwords
                return m

            else:
                return None

        mock_open.side_effect = open_fake_file

        def bz2file_fake(file_name, state):
            m = MagicMock()
            m.__enter__.return_value = Mock()
            m.__exit__.return_value = Mock()
            m.__enter__.return_value = file_name
            return m

        mock_bz2file.side_effect = bz2file_fake

        data_source_name = 'dummy.pkl.bz2'
        tfidf_file_name = os.path.join('outputs', 'tfidf', data_source_name + '-tfidf.pkl.bz2')
        args = ['-o', 'tfidf', '-ds', data_source_name, '--id_header', 'patent_id', '--date_header', 'publication_date',
                '--max_document_frequency', '1.0']

        return_value = pygrams.main(args)

        self.assertEqual(0, return_value, 'Return value indicates failure')

        results_checked = False
        for dump_args in mock_pickle_dump.call_args_list:
            if dump_args[0][1] == tfidf_file_name:
                [tfidf_matrix, feature_names, document_week_dates, doc_ids] = dump_args[0][0]

                self.assertEqual(tfidf_matrix.todense(), np.ones(shape=(1, 1)), 'TFIDF should be 1x1 matrix of 1')
                self.assertListEqual(feature_names, ['abstract'])
                self.assertListEqual(document_week_dates, [199901])
                self.assertListEqual(doc_ids, ['family0'])
                results_checked = True
                break

        if not results_checked:
            self.fail('Results were not matched - were filenames correct?')


if __name__ == '__main__':
    unittest.main()
