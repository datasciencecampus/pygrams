import pandas as pd
import unittest

from scripts import FilePaths
from scripts.algorithms.tfidf import TFIDF, StemTokenizer
from scripts.visualization.graphs.terms_graph import TermsGraph


class TestGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_ngrams = 50
        df = pd.read_pickle(FilePaths.us_patents_random_1000_pickle_name)
        tfidf_obj = TFIDF(df, ngram_range=(2, 4), tokenizer=StemTokenizer(), normalize_doc_length=False)
        terms, freqs_list = tfidf_obj.detect_popular_ngrams_in_docs_set( number_of_ngrams_to_return = num_ngrams)
        graph_obj = TermsGraph(freqs_list, tfidf_obj)
        graph = graph_obj.graph
        cls.__links = graph['links']
        cls.__nodes = graph['nodes']

    def test_num_nodes(self):
        self.assertEquals(50, len(self.__nodes))

    def test_num_links(self):
        self.assertEquals(471, len(self.__links))

    def test_terms_in_nodes(self):
        texts = [x['text'] for x in self.__nodes]

        self.assertIn('central portion', texts)
        self.assertIn('fluid commun', texts)
        self.assertIn('provid seed', texts)
        self.assertIn('gate line', texts)

        idx_1 = texts.index("central portion")
        idx_2 = texts.index("fluid commun")
        idx_3 = texts.index("provid seed")
        idx_4 = texts.index("gate line")

        self.assertAlmostEqual(0.054940822918518054, self.__nodes[idx_1]['freq'])
        self.assertAlmostEqual(0.032458775009343,    self.__nodes[idx_2]['freq'])
        self.assertAlmostEqual(0.014660739181626243, self.__nodes[idx_3]['freq'])
        self.assertAlmostEqual(0.07741770854955801,  self.__nodes[idx_4]['freq'])

    def test_terms_in_links(self):

        texts = [(x['source'], x['target']) for x in self.__links]

        self.assertIn(('semiconductor substrat', 'diffus barrier materi'), texts)
        self.assertIn(('lock arm', 'swing arm'), texts)
        self.assertIn(('bodi portion', 'elastomer portion'), texts)
        self.assertIn(('central portion', 'convex outer surfac'), texts)

        idx_1 = texts.index(('semiconductor substrat', 'diffus barrier materi'))
        idx_2 = texts.index(('lock arm', 'swing arm'))
        idx_3 = texts.index(('bodi portion', 'elastomer portion'))
        idx_4 = texts.index(('central portion', 'convex outer surfac'))

        self.assertAlmostEqual(0.179044009721727, self.__links[idx_1]['size'])
        self.assertAlmostEqual(0.0969491167354179,    self.__links [idx_2]['size'])
        self.assertAlmostEqual(0.21718768050108705, self.__links [idx_3]['size'])
        self.assertAlmostEqual(0.11142954674438521,  self.__links[idx_4]['size'])




