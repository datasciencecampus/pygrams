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
        'sample term 100% correct': {5: {
            0: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            1: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            2: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            'accuracy': 3 / 3}
        },
        'second term 67% correct': {5: {
            0: {'predicted_label': 't-decrease',
                'label': 't-increase'},
            1: {'predicted_label': 't-decrease',
                'label': 't-decrease'},
            2: {'predicted_label': 't-decrease',
                'label': 't-decrease'},
            'accuracy': 2 / 3}
        },
        'third term 33% correct': {5: {
            0: {'predicted_label': 't-increase',
                'label': 'p-increase'},
            1: {'predicted_label': 'p-increase',
                'label': 't-increase'},
            2: {'predicted_label': 't-increase',
                'label': 't-increase'},
            'accuracy': 1 / 3}
        },
        'fourth term 100% correct': {5: {
            0: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            1: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            2: {'predicted_label': 'p-increase',
                'label': 'p-increase'},
            'accuracy': 3 / 3}
        },
        'fifth term 0% correct': {5: {
            0: {'predicted_label': 'p-decrease',
                'label': 'p-increase'},
            1: {'predicted_label': 'p-increase',
                'label': 'p-decrease'},
            2: {'predicted_label': 'p-increase',
                'label': 'p-decrease'},
            'accuracy': 0 / 3}
        }
    }

    def test_html_table(self):
        expected_text = '''Terms 5
sample term 100% correct 100%
second term 67% correct 67%
third term 33% correct 33%
fourth term 100% correct 100%
fifth term 0% correct 0%
'''

        output_html = html_table(self.example_results)

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)

    def test_summary_html_table(self):
        expected_text = '''Summary 5
Mean 60%
Trimmed (20% cut) mean 67%
Summary 5
Standard deviation 43%
Trimmed (20% cut) standard deviation 33%
'''

        output_html = summary_html_table(self.example_results, trimmed_proportion_to_cut=0.2)

        soup = BeautifulSoup(output_html, 'html.parser')

        actual_text = extract_table_text_from_html(soup)
        self.assertEqual(expected_text, actual_text)
