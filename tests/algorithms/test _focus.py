import unittest
import app_detect.scripts.algorithms.text_stats as ts

from app_detect.scripts.algorithms.tf_idf import TFIDF, LemmaTokenizer
from app_detect.scripts.data_access.patent_data_factory import PatentDataFactory


class TestFocus(unittest.TestCase):

    def test_popular_ngrams_by_chi2_importance(self):
        expected_output = [
        'control system',
        'exhaust gas',
        'control unit',
        'control device',
        'solar panel',
        'internal combustion',
        'internal combustion engine',
        ]

        textstats = ts.TextStats()
        num_ngrams = 25
        df = PatentDataFactory.getdf("green_growth_1000", datefrom='2000-01-01', dateto='2018-01-01')
        tfidf = TFIDF(df, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))

        newdf = PatentDataFactory.getdf("US_non_Y02_1000", datefrom='2000-01-01', dateto='2018-01-01')
        tfidf2 = TFIDF(newdf, tokenizer=LemmaTokenizer(), ngram_range=(2, 3))

        actual_output = focus.popular_ngrams_by_chi2_importance(tfidf, tfidf2, num_ngrams)

        self.assertListEqual(expected_output, actual_output)



