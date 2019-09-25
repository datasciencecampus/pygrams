import unittest

from bs4 import BeautifulSoup

from scripts.vandv.ssm_reporting import html_table, summary_html_table


def extract_table_text_from_html(soup):
    extracted_text = ''
    for tag in soup.children:
        if tag.name == 'table':
            table_text = tag.text.replace('\n\n', '§').replace('\n', ' ').replace('§', '\n').strip()
            table_lines = [x.strip() for x in table_text.split('\n')]
            extracted_text += '\n'.join(table_lines) + '\n'
    return extracted_text


class SSMReporting(unittest.TestCase):
    example_results = {
        'sample term 100% correct': {'ARIMA_3': 3 / 3, 'ARIMA_5': 0.8},
        'second term 67% correct': {'ARIMA_3': 2 / 3, 'ARIMA_5': 0.8},
        'third term 33% correct': {'ARIMA_3': 1 / 3, 'ARIMA_5': 0.8},
        'fourth term 100% correct': {'ARIMA_3': 3 / 3, 'ARIMA_5': 0.8},
        'fifth term 0% correct': {'ARIMA_3': 0 / 3, 'ARIMA_5': 0.8},
    }

    def test_html_table(self):
        expected_text = '''Terms ARIMA_3 ARIMA_5
sample term 100% correct 100% 80%
second term 67% correct 67% 80%
third term 33% correct 33% 80%
fourth term 100% correct 100% 80%
fifth term 0% correct 0% 80%
'''

        output_html = html_table(self.example_results)

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)

    def test_summary_html_table(self):
        expected_text = '''Summary ARIMA_3 ARIMA_5
Mean 60% 80%
Trimmed (20% cut) mean 67% 80%
Summary ARIMA_3 ARIMA_5
Standard deviation 43% 0%
Trimmed (20% cut) standard deviation 33% 0%
'''

        output_html = summary_html_table(self.example_results, trimmed_proportion_to_cut=0.2)

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)
