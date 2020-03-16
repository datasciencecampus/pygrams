import os
import unittest

import numpy as np
import pandas as pd

from scripts import FilePaths
from scripts.filter_terms import FilterTerms
from scripts.terms_graph import TermsGraph
from scripts.text_processing import StemTokenizer
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import tfidf_from_text
from scripts.utils import utils


class TestGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_ngrams = 50
        min_n = 2
        max_n = 3
        max_df=0.3
        ngram_range = (min_n, max_n)

        df = pd.read_pickle(os.path.join('..', FilePaths.us_patents_random_1000_pickle_name))
        tfidf_obj = tfidf_from_text(df['abstract'], ngram_range=ngram_range, max_document_frequency=max_df,
                                    tokenizer=StemTokenizer())

        doc_weights = list(np.ones(len(df)))

        # term weights - embeddings
        filter_output_obj = FilterTerms(tfidf_obj.feature_names, None, None)
        term_weights = filter_output_obj.ngram_weights_vec

        tfidf_mask_obj = TfidfMask(tfidf_obj, ngram_range=ngram_range, unbias=True)
        tfidf_mask_obj.update_mask(doc_weights, term_weights)
        tfidf_mask = tfidf_mask_obj.tfidf_mask

        # mask the tfidf matrix
        tfidf_matrix = tfidf_obj.tfidf_matrix
        tfidf_masked = tfidf_mask.multiply(tfidf_matrix)
        tfidf_masked = utils.remove_all_null_rows(tfidf_masked)

        print(f'Processing TFIDF matrix of {tfidf_masked.shape[0]:,} / {tfidf_matrix.shape[0]:,} documents')

        cls.__tfidf_reduce_obj = TfidfReduce(tfidf_masked, tfidf_obj.feature_names)
        term_score_tuples = cls.__tfidf_reduce_obj.extract_ngrams_from_docset('sum')
        graph_obj = TermsGraph(term_score_tuples[:num_ngrams], cls.__tfidf_reduce_obj)
        graph = graph_obj.graph
        cls.__links = graph['links']
        cls.__nodes = graph['nodes']

    def test_num_nodes(self):
        self.assertEqual(50, len(self.__nodes))

    def test_num_links(self):
        self.assertEqual(182, len(self.__links))

    def test_terms_in_nodes(self):
        texts = [x['text'] for x in self.__nodes]

        term1 = 'liquid crystal display'
        term2 = 'light emit diod'
        term3 = 'silicon oxid film'
        term4 = 'memori cell array'

        self.assertIn(term1, texts)
        self.assertIn(term2, texts)
        self.assertIn(term3, texts)
        self.assertIn(term4, texts)

        idx_1 = texts.index(term1)
        idx_2 = texts.index(term2)
        idx_3 = texts.index(term3)
        idx_4 = texts.index(term4)

        self.assertAlmostEqual(1.0,  self.__nodes[idx_1]['freq'])
        self.assertAlmostEqual(0.4157084570362393,   self.__nodes[idx_2]['freq'])
        self.assertAlmostEqual(0.171528689949101, self.__nodes[idx_3]['freq'])
        self.assertAlmostEqual(0.19875770822081537,  self.__nodes[idx_4]['freq'])

    def test_terms_in_links(self):

        texts = [(x['source'], x['target']) for x in self.__links]

        link_1 = ('intern combust engin', 'project cylind head')
        link_2 = ('treatment compound', 'inhibit protein kinas')
        link_3 = ('composit compound', 'inhibit aak1')
        link_4 = ('pharmaceut accept salt', 'activ hepat viru')

        self.assertIn(link_1, texts)
        self.assertIn(link_2, texts)
        self.assertIn(link_3, texts)
        self.assertIn(link_4, texts)

        idx_1 = texts.index(link_1)
        idx_2 = texts.index(link_2)
        idx_4 = texts.index(link_3)
        idx_3 = texts.index(link_4)

        self.assertAlmostEqual(0.3813920347587772, self.__links[idx_1]['size'])
        self.assertAlmostEqual(0.7144144576377361, self.__links[idx_2]['size'])
        self.assertAlmostEqual(0.44907177942611987, self.__links[idx_3]['size'])
        self.assertAlmostEqual(0.5741197010309195, self.__links[idx_4]['size'])
