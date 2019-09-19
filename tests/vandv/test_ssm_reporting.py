import unittest

from bs4 import BeautifulSoup

from scripts.vandv.ssm_reporting import html_table, summary_html_table


def extract_table_text_from_html(soup):
    for tag in soup.children:
        if tag.name == 'table':
            table_text = tag.text.replace('\n\n', '§').replace('\n', ' ').replace('§', '\n').strip()
            table_lines = [x.strip() for x in table_text.split('\n')]
            return '\n'.join(table_lines)
    return None


class SSMReporting(unittest.TestCase):
    def test_html_table(self):
        results = {
            'sample term': {2: 78, 3: 60},
            'extra term': {2: 93, 3: 87}
        }
        expected_text = '''terms 2 3
sample term 78 60
extra term 93 87'''

        output_html = html_table(results, [2, 3])

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)

    def test_summary_html_table(self):
        results = {
            'sample term': {2: 80, 3: 60},
            'extra term': {2: 90, 3: 60},
            'third term': {2: 100, 3: 60},
            'fourth term': {2: 1000, 3: -200},
            'fifth term': {2: -2000, 3: 600}
        }
        expected_text = '''terms 2 3
Mean -146 116
Trimmed (20% cut) mean 90 60
Standard deviation 1108.82 293.053
Trimmed (20% cut) standard deviation 10 0'''

        output_html = summary_html_table(results, [2, 3], trimmed_proportion_to_cut=0.2)

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)
