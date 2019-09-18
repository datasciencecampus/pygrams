import unittest

from scripts.vandv.ssm_reporting import html_table


class SSMReporting(unittest.TestCase):
    def test_html_table(self):
        results = {
            'sample term': {2: 78, 3: 60}
        }

        output_html = html_table(results, [2, 3])

        self.assertEqual(
            '''
            '''
            , output_html
        )
