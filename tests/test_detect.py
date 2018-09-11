import os
import unittest
from unittest import mock

import detect


class TestDetect(unittest.TestCase):

    def test_args_json_not_requested(self):
        args = detect.get_args([])
        self.assertFalse(args.json)

    def test_args_json_requested_short(self):
        args = detect.get_args(['-j'])
        self.assertTrue(args.json)

    def test_args_json_requested_long(self):
        args = detect.get_args(['--json'])
        self.assertTrue(args.json)

    def test_args_report_name_requested_long(self):
        args = detect.get_args(['--report_name=my/test/name.txt'])
        self.assertEqual('my/test/name.txt', args.report_name)

    def test_args_patent_source_requested_long(self):
        args = detect.get_args(['--patent_source=my-test'])
        self.assertEqual('my-test', args.patent_source)

    @mock.patch("detect.json.dump", create=True)
    @mock.patch("detect.open", create=True)
    def test_json_configuration_encoding_minimal(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'test.pkl'
        report_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.txt')
        json_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.json')
        args = detect.get_args(['-j', f'--report_name={report_file_name}'])

        detect.write_config_to_json(args, patent_pickle_file_name)

        self.assertTrue(args.json)
        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_file_name, 'tech_report': report_file_name
            },
            'year': {
                'from': 2000,
                'to': 0
            },
            'parameters': {
                'cite': False,
                'cpc': '',
                'focus': False,
                'pick': 'sum',
                'time': False
            }
        }
        self.assertEqual(expected_json, actual_json)

    @mock.patch("detect.json.dump", create=True)
    @mock.patch("detect.open", create=True)
    def test_json_configuration_encoding_maximal(self, mock_open, mock_json_dump):
        patent_pickle_file_name = 'test.pkl'
        report_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.txt')
        json_file_name = os.path.join(os.path.abspath(os.sep), 'dummy', 'test.json')
        args = detect.get_args(['-j', f'--report_name={report_file_name}', '-c', '-t', '-f', '-p=max', '-cpc=Y12',
                                '-yf=1998', '-yt=2001'])

        detect.write_config_to_json(args, patent_pickle_file_name)

        self.assertTrue(args.json)
        mock_open.assert_called_with(json_file_name, 'w')

        actual_json = mock_json_dump.call_args[0][0]
        expected_json = {
            'paths': {
                'data': patent_pickle_file_name, 'tech_report': report_file_name
            },
            'year': {
                'from': 1998,
                'to': 2001
            },
            'parameters': {
                'cite': True,
                'cpc': 'Y12',
                'focus': True,
                'pick': 'max',
                'time': True
            }
        }
        self.assertEqual(expected_json, actual_json)


if __name__ == '__main__':
    unittest.main()
